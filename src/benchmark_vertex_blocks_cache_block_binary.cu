/*!

b_vertex_t_edge_binary

One thread-block per src
One thread per src-dst edge
Binary search of dst neighbor list into src list

*/

#include <fmt/format.h>
#include <iostream>

#include <nvToolsExt.h>

#include "clara/clara.hpp"

#include "pangolin/configure.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"

#include "pangolin/algorithm/tc_vertex_blocks_cache_block_binary.cuh"
#include "pangolin/sparse/csr.hpp"

int main(int argc, char **argv) {

  pangolin::init();

  std::vector<int> gpus;
  std::string path;
  int iters = 1;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  bool readMostly = false;
  bool accessedBy = false;
  bool prefetchAsync = false;

  bool upperTriangular = false;
  int rowCacheSz = 512;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(gpus, "dev ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(readMostly)["--read-mostly"]("mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(accessedBy)["--accessed-by"]("mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(prefetchAsync)["--prefetch-async"]("prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(iters, "N")["-n"]("number of counts");
  cli =
      cli | clara::Opt(upperTriangular)["--upper-triangular"]("convert to DAG by dst > src (default lower-triangular)");
  cli = cli | clara::Opt(rowCacheSz, "INT")["-r"]("Size of shared-memory row cache");
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

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  LOG(info, "using {}B ints", sizeof(uint32_t));
  std::vector<pangolin::DiEdge<uint32_t>> edges;
  std::vector<pangolin::DiEdge<uint32_t>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;
  for (int i = 0; i < iters; ++i) {
    // create csr
    nvtxRangePush("create csr");
    start = std::chrono::system_clock::now();
    auto upperTriangularFilter = [](pangolin::DiEdge<uint32_t> e) { return e.src < e.dst; };
    auto lowerTriangularFilter = [](pangolin::DiEdge<uint32_t> e) { return e.src > e.dst; };
    auto csr = upperTriangular ? pangolin::CSR<uint32_t>::from_edges(edges.begin(), edges.end(), upperTriangularFilter)
                               : pangolin::CSR<uint32_t>::from_edges(edges.begin(), edges.end(), lowerTriangularFilter);

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop(); // create csr
    LOG(debug, "nnz = {}", csr.nnz());
    LOG(info, "create CSR time {}s", elapsed);

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

    // accessed-by
    nvtxRangePush("accessed-by");
    start = std::chrono::system_clock::now();
    if (accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "accessed-by CSR time {}s", elapsed);

    // prefetch
    nvtxRangePush("prefetch");
    start = std::chrono::system_clock::now();
    if (prefetchAsync) {
      for (const auto &gpu : gpus) {
        csr.prefetch_async(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "prefetch CSR time {}s", elapsed);

    // count triangles
    nvtxRangePush("count");
    start = std::chrono::system_clock::now();

    // create async counters
    std::vector<pangolin::VertexBlocksCacheBlockBinary> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(pangolin::VertexBlocksCacheBlockBinary(dev, rowCacheSz));
    }

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "ctor time {}s", elapsed);

    // determine the number of rows per GPU
    const size_t rowsPerGPU = (csr.num_rows() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} rows per GPU", rowsPerGPU);

    // launch counting operations
    size_t rowStart = 0;
    for (auto &counter : counters) {
      const size_t rowStop = std::min(rowStart + rowsPerGPU, csr.num_rows());
      const size_t numRows = rowStop - rowStart;
      LOG(debug, "start async count on GPU {} ({} rows)", counter.device(), numRows);
      counter.count_async(csr.view(), rowStart, numRows);
      rowStart += rowsPerGPU;
    }

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "launch time {}s", elapsed);

    // wait for counting operations to finish
    uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop(); // count
    LOG(info, "count time {}s", elapsed);
    LOG(info, "{} triangles ({} teps)", total, csr.nnz() / elapsed);
    times.push_back(elapsed);
    tris = total;
    nnz = csr.nnz();
  }

  std::cout << path << ",\t" << nnz << ",\t" << tris;
  for (const auto &t : times) {
    std::cout << ",\t" << t;
  }
  std::cout << std::endl;

  return 0;
}
