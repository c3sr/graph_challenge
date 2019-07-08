/*!

Count triangles using the per-edge binary search

*/

#include <iostream>

#include <nvToolsExt.h>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include "pangolin/algorithm/tc_edge_binary.cuh"
#include "pangolin/configure.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_coo.hpp"

struct RunOptions {
  std::string path; //!< path for graph
  std::string sep;  //!< seperator for output
  std::vector<int> gpus;
  int blockSize;
  int coarsening;
  int iters;

  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  bool shrinkToFit;
};

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

template <typename Index> int run(RunOptions &opts) {
  typedef typename pangolin::EdgeTy<Index> Edge;

  auto gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(opts.path);

  std::vector<Edge> edges;
  std::vector<Edge> fileEdges;
  while (file.get_edges(fileEdges, 500)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // record various times for each iteration
  std::vector<double> iterationTimes(opts.iters, 0);   // the total wall time elapsed during each iteration
  std::vector<double> competitionTimes(opts.iters, 0); // the actual competition time
  std::vector<double> csrTimes(opts.iters, 0);         // time taken to build the CSR
  std::vector<double> kernelTimes(opts.iters,
                                  0); // just the triangle counting kernel
  std::vector<double> countTimes(
      opts.iters, 0); // the wall time elapsed configuring, launching, and waiting for counting operations
  std::vector<double> readMostlyTimes(opts.iters,
                                      0);           // times elapsed during read-mostly
  std::vector<double> prefetchTimes(opts.iters, 0); // times elapsed during prefetch
  std::vector<double> accessedByTimes(opts.iters,
                                      0); // times elapsed during accessed-by
  std::vector<double> counterCtorTimes(opts.iters,
                                       0); // time taken to construct counters

  uint64_t nnz;
  uint64_t tris;
  uint64_t numRows;
  // create csr and count `opts.iters` times
  for (int i = 0; i < opts.iters; ++i) {
    auto iterStart = std::chrono::system_clock::now();
    auto competitionStart = std::chrono::system_clock::now();
    // create csr
    auto upperTriangularFilter = [](Edge e) { return e.first < e.second; };
    auto lowerTriangularFilter = [](Edge e) { return e.first > e.second; };
    auto csr = pangolin::CSRCOO<uint64_t>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);

    if (opts.shrinkToFit) {
      LOG(debug, "shrink CSR");
      csr.shrink_to_fit();
    }

    elapsed = (std::chrono::system_clock::now() - iterStart).count() / 1e9;
    LOG(debug, "CSR nnz = {} rows = {}", csr.nnz(), csr.num_rows());
    LOG(debug, "CSR cap = {}MB size = {}MB", csr.capacity_bytes() / 1024 / 1024, csr.size_bytes() / 1024 / 1024);
    LOG(info, "create CSR time {}s", elapsed);
    csrTimes[i] = elapsed;

    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (opts.readMostly) {
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
    if (opts.accessedBy) {
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
    if (opts.prefetchAsync) {
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
      counter.count_async(csr.view(), numEdges, edgeStart, opts.blockSize, opts.coarsening);
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
    if (counters.size() == 1) {
      kernelTimes[i] = counters[0].kernel_time();
    } else {
      kernelTimes[i] = 0;
    }

    tris = total;
    nnz = csr.nnz();
    numRows = csr.num_rows();
  }

  if (opts.iters > 0) {
    fmt::print("binary");
    fmt::print("{}{}", opts.sep, opts.blockSize);
    std::string gpuStr;
    for (auto gpu : gpus) {
      gpuStr += std::to_string(gpu);
    }
    fmt::print("{}{}", opts.sep, gpuStr);
    fmt::print("{}{}", opts.sep, opts.path);
    fmt::print("{}{}", opts.sep, numRows);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, tris);

    print_vec(readMostlyTimes, opts.sep);
    print_vec(accessedByTimes, opts.sep);
    print_vec(prefetchTimes, opts.sep);

    print_vec(countTimes, opts.sep);
    for (const auto &s : countTimes) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
    print_vec(kernelTimes, opts.sep);
    for (const auto &s : kernelTimes) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
    fmt::print("\n");
  }

  return 0;
}

void print_header(const RunOptions &opts) {
  fmt::print("bmark{0}bs{0}graph{0}nodes{0}edges{0}tris", opts.sep);
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}readMostly{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}accessedBy{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}prefetchAsync{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}time{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}teps{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}kernel_time{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}kernel_teps{}", opts.sep, i);
  }
  fmt::print("\n");
}

int main(int argc, char **argv) {

  pangolin::init();

  RunOptions opts;
  opts.sep = ",";
  opts.blockSize = 512;
  opts.coarsening = 1;
  opts.iters = 1;
  opts.shrinkToFit = false;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool onlyPrintHeader = false;
  bool wide = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(onlyPrintHeader)["--header"]("print the header for the times output and quit");
  cli = cli | clara::Opt(wide)["--wide"]("64-bit node IDs");
  cli = cli | clara::Opt(opts.gpus, "dev ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(opts.coarsening, "coarsening")["-c"]("Number of elements per thread");
  cli = cli | clara::Opt(opts.blockSize, "block-dim")["--bs"]("Number of threads in a block");
  cli = cli | clara::Opt(opts.shrinkToFit)["--shrink-to-fit"]("shrink allocations to fit data");
  cli = cli | clara::Opt(opts.readMostly)["--read-mostly"]("mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(opts.accessedBy)["--accessed-by"]("mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(opts.prefetchAsync)["--prefetch-async"]("prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(opts.iters, "N")["-n"]("number of counts");
  cli = cli | clara::Arg(opts.path, "graph file")("Path to adjacency list").required();

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

  if (onlyPrintHeader) {
    print_header(opts);
    return 0;
  }
  if (wide) {
    return run<uint64_t>(opts);
  } else {
    return run<uint32_t>(opts);
  }
}
