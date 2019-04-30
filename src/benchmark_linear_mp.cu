/*!

Count triangles using the per-edge linear search.
Use one thread for each triangle counter through OpenMP.

*/

#include <fmt/format.h>
#include <iostream>

#include <nvToolsExt.h>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {

  pangolin::init();

  pangolin::Config config;

  std::vector<int> gpus;
  std::string path;
  int iters = 1;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  bool readMostly = false;
  bool accessedBy = false;
  bool prefetchAsync = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(gpus, "ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(readMostly)["--read-mostly"](
                  "mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(accessedBy)["--accessed-by"](
                  "mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(prefetchAsync)["--prefetch-async"](
                  "prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(iters, "N")["-n"]("number of counts");
  cli =
      cli | clara::Arg(path, "graph file")("Path to adjacency list").required();

  auto result = cli.parse(clara::Args(argc, argv));
  if (!result) {
    LOG(error, "Error in command line: {}", result.errorMessage());
    exit(1);
  }

  if (help) {
    std::cout << cli;
    return 0;
  }

  // set logging level
  if (verbose) {
    pangolin::logger::set_level(pangolin::logger::Level::TRACE);
  } else if (debug) {
    pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  }

  // log command line before much else happens
  {
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
      if (i != 0) {
        cmd += " ";
      }
      cmd += argv[i];
    }
    LOG(debug, cmd);
  }
  LOG(debug, "pangolin version: {}.{}.{}", PANGOLIN_VERSION_MAJOR,
      PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
  LOG(debug, "pangolin branch:  {}", PANGOLIN_GIT_REFSPEC);
  LOG(debug, "pangolin sha:     {}", PANGOLIN_GIT_HASH);
  LOG(debug, "pangolin changes: {}", PANGOLIN_GIT_LOCAL_CHANGES);

#ifndef NDEBUG
  LOG(warn, "Not a release build");
#endif

  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // Check for unified memory support
  bool managed = true;
  cudaDeviceProp prop;
  for (auto gpu : gpus) {
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, gpu));
    // We check for concurrentManagedAccess, as devices with only the
    // managedAccess property have extra synchronization requirements.
    if (!prop.concurrentManagedAccess) {
      LOG(warn, "device {} does not support concurrentManagedAccess", gpu);
    }
    managed = managed && prop.concurrentManagedAccess;
  }

  if (managed) {
    LOG(debug, "managed memory supported");
  } else {
    LOG(warn, "managed memory not supported!");
  }

  // set up GPUs
  nvtxRangePush("setup");
  for (auto gpu : gpus) {
    CUDA_RUNTIME(cudaSetDevice(gpu));
    CUDA_RUNTIME(cudaFree(0));
  }
  nvtxRangePop();

  // read data
  nvtxRangePush("read data");
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  std::vector<pangolin::EdgeTy<uint64_t>> edges;
  std::vector<pangolin::EdgeTy<uint64_t>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  nvtxRangePop();
  LOG(debug, "read {} edges", edges.size());

  // create one stream per GPU
  nvtxRangePush("create streams");
  std::vector<cudaStream_t> streams(gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i) {
    CUDA_RUNTIME(cudaSetDevice(gpus[i]));
    CUDA_RUNTIME(cudaStreamCreate(&streams[i]));
  }
  nvtxRangePop();

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;
  for (int i = 0; i < iters; ++i) {

    // create csr
    CUDA_RUNTIME(cudaSetDevice(gpus[0]));
    nvtxRangePush("create CSR");
    start = std::chrono::system_clock::now();
    auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
      return e.first < e.second;
    };
    auto csr = pangolin::COO<uint64_t>::from_edges(edges.begin(), edges.end(),
                                                   upperTriangular);
    nvtxRangePop();
    LOG(debug, "nnz = {}", csr.nnz());
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "create CSR time {}s", elapsed);

    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (readMostly) {
      csr.read_mostly();
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);

    // accessed-by
    nvtxRangePush("accessed-by");
    start = std::chrono::system_clock::now();
    if (accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "accessed-by CSR time {}s", elapsed);

    uint64_t total = 0; // total triangle count

    omp_set_num_threads(gpus.size());
    start = std::chrono::system_clock::now();
#pragma omp parallel for
    for (size_t gpuIdx = 0; gpuIdx < gpus.size(); ++gpuIdx) {
      const int gpu = gpus[gpuIdx];
      cudaStream_t stream = streams[gpuIdx];
      CUDA_RUNTIME(cudaSetDevice(gpu));

      // prefetch
      if (prefetchAsync) {
        LOG(debug, "omp thread {}: prefetch csr to device {}",
            omp_get_thread_num(), gpu);
        nvtxRangePush("prefetch");
        csr.prefetch_async(gpu, stream);
        nvtxRangePop();
      }

      // count triangles
      nvtxRangePush("count");

      // create async counters
      LOG(debug, "omp thread {}: create device {} counter",
          omp_get_thread_num(), gpu);
      pangolin::LinearTC counter(gpu, stream);

      // determine the number of edges per gpu
      const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
      LOG(debug, "omp thread {}: {} edges per GPU", omp_get_thread_num(),
          edgesPerGPU);

      // launch counting operations
      const size_t edgeStart = edgesPerGPU * gpuIdx;
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(),
          numEdges);
      counter.count_async(csr.view(), numEdges, edgeStart);

      // wait for counting operations to finish
      LOG(debug, "omp thread {}: wait for counter on GPU {}",
          omp_get_thread_num(), counter.device());
      counter.sync();
      nvtxRangePop();
#pragma omp atomic
      total += counter.count();
    } // gpus
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "hints/count time {}s", elapsed);
    LOG(info, "{} triangles ({} teps)", total, csr.nnz() / elapsed);
    times.push_back(elapsed);
    tris = total;
    nnz = csr.nnz();

  } // iters

  for (auto stream : streams) {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
  }

  std::cout << path << ",\t" << nnz << ",\t" << tris;
  for (const auto &t : times) {
    std::cout << ",\t" << t;
  }
  std::cout << std::endl;

  return 0;
}
