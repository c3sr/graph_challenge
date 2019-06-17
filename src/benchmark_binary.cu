/*!

Count triangles using the per-edge binary search

*/

#include <fmt/format.h>
#include <iostream>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {

  pangolin::init();

  std::string sep(",");

  std::vector<int> gpus;
  std::string path;
  int dimBlock = 512;
  int coarsening = 1;
  int iters = 1;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  bool onlyPrintHeader = false;

  bool readMostly = false;
  bool accessedBy = false;
  bool prefetchAsync = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(onlyPrintHeader)["--header"]("print the header for the times output and quit");
  cli = cli | clara::Opt(gpus, "dev ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(coarsening, "coarsening")["-c"]("Number of elements per thread");
  cli = cli | clara::Opt(dimBlock, "block-dim")["--bs"]("Number of threads in a block");
  cli = cli | clara::Opt(readMostly)["--read-mostly"]("mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(accessedBy)["--accessed-by"]("mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(prefetchAsync)["--prefetch-async"]("prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(iters, "N")["-n"]("number of counts");
  cli = cli | clara::Arg(path, "graph file")("Path to adjacency list").required();

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
  LOG(debug, "pangolin version: {}.{}.{}", PANGOLIN_VERSION_MAJOR, PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
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

  if (onlyPrintHeader) {
    std::cout << "path" << sep << "nnz" << sep << "tris" << sep << "bs";
    for (int i = 0; i < iters; ++i) {
      std::cout << sep << "count" + std::to_string(i);
    }
    for (int i = 0; i < iters; ++i) {
      std::cout << sep << "competition" + std::to_string(i);
    }
    for (int i = 0; i < iters; ++i) {
      std::cout << sep << "iter" + std::to_string(i);
    }

    std::cout << std::endl;
    return 0;
  }

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  std::vector<pangolin::EdgeTy<uint64_t>> edges;
  std::vector<pangolin::EdgeTy<uint64_t>> fileEdges;
  while (file.get_edges(fileEdges, 500)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // record various times for each iteration
  std::vector<double> iterationTimes(iters, 0);   // the total wall time elapsed during each iteration
  std::vector<double> competitionTimes(iters, 0); // the actual competition time
  std::vector<double> csrTimes(iters, 0);         // time taken to build the CSR
  std::vector<double> kernelTimes(iters,
                                  0); // just the triangle counting kernel
  std::vector<double> countTimes(
      iters, 0); // the wall time elapsed configuring, launching, and waiting for counting operations
  std::vector<double> readMostlyTimes(iters,
                                      0);      // times elapsed during read-mostly
  std::vector<double> prefetchTimes(iters, 0); // times elapsed during prefetch
  std::vector<double> accessedByTimes(iters,
                                      0); // times elapsed during accessed-by
  std::vector<double> counterCtorTimes(iters,
                                       0); // time taken to construct counters

  uint64_t nnz;
  uint64_t tris;
  // create csr and count `iters` times
  for (int i = 0; i < iters; ++i) {
    auto iterStart = std::chrono::system_clock::now();
    auto competitionStart = std::chrono::system_clock::now();
    // create csr
    auto upperTriangularFilter = [](pangolin::EdgeTy<uint64_t> e) { return e.first < e.second; };
    auto lowerTriangularFilter = [](pangolin::EdgeTy<uint64_t> e) { return e.first > e.second; };
    auto csr = pangolin::COO<uint64_t>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);
    LOG(debug, "CSR nnz = {} rows = {}", csr.nnz(), csr.num_rows());
    elapsed = (std::chrono::system_clock::now() - iterStart).count() / 1e9;
    LOG(info, "create CSR time {}s", elapsed);
    csrTimes[i] = elapsed;

    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (readMostly) {
      csr.read_mostly();
      for (const auto &gpu : gpus) {
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);
    readMostlyTimes[i] = elapsed;

    // accessed-by
    start = std::chrono::system_clock::now();
    if (accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "accessed-by CSR time {}s", elapsed);
    accessedByTimes[i] = elapsed;

    // prefetch
    start = std::chrono::system_clock::now();
    if (prefetchAsync) {
      for (const auto &gpu : gpus) {
        csr.prefetch_async(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "prefetch CSR time {}s", elapsed);
    prefetchTimes[i] = elapsed;

    // create async counters
    const auto counterCtorStart = std::chrono::system_clock::now();
    std::vector<pangolin::BinaryTC> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(std::move(pangolin::BinaryTC(dev)));
    }
    elapsed = (std::chrono::system_clock::now() - counterCtorStart).count() / 1e9;
    counterCtorTimes[i] = elapsed;

    const auto countStart = std::chrono::system_clock::now();

    // determine the number of edges per gpu
    const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    size_t edgeStart = 0;
    for (auto &counter : counters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(), numEdges);
      counter.count_async(csr.view(), numEdges, edgeStart, dimBlock, coarsening);
      edgeStart += edgesPerGPU;
    }

    // wait for counting operations to finish
    uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }

    const auto countStop = std::chrono::system_clock::now();

    elapsed = (countStop - countStart).count() / 1e9;
    LOG(info, "count time {}s", elapsed);
    LOG(info, "{} triangles ({} teps)", total, csr.nnz() / elapsed);
    countTimes[i] = elapsed;
    elapsed = (countStop - competitionStart).count() / 1e9;
    competitionTimes[i] = elapsed;
    elapsed = (countStop - iterStart).count() / 1e9;
    iterationTimes[i] = elapsed;

    tris = total;
    nnz = csr.nnz();
  }

  std::cout << path << sep << nnz << sep << tris << sep << dimBlock;
  for (auto t : countTimes) {
    std::cout << sep << t;
  }
  for (auto t : competitionTimes) {
    std::cout << sep << t;
  }
  for (auto t : iterationTimes) {
    std::cout << sep << t;
  }

  std::cout << std::endl;

  return 0;
}
