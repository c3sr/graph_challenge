/*!

Count triangles using warp-granularity dynamic algorithm selection

*/

#include <iostream>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include <fmt/format.h>

#include "pangolin/algorithm/tc_vertex_dyn.cuh"
#include "pangolin/bounded_buffer.hpp"
#include "pangolin/configure.hpp"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr.hpp"

struct RunOptions {
  std::vector<int> gpus;
  std::string path;
  std::string sep;
  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  int blockSize;
  int iters;
};

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

void print_header(RunOptions &opts) {
  fmt::print("bmark{0}bs{0}graph{0}nodes{0}edges{0}tris", opts.sep);

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

template <typename Index> int run(RunOptions &opts) {
  typedef pangolin::EdgeTy<Index> Edge;

  std::vector<int> gpus = opts.gpus;
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

  // create csr and count `iters` times
  std::vector<double> times;
  std::vector<double> kernelTimes;
  uint64_t nodes;
  uint64_t tris;
  uint64_t nnz;
  for (int i = 0; i < opts.iters; ++i) {
    // create csr
    start = std::chrono::system_clock::now();
    auto upperTriangularFilter = [](Edge e) { return e.first < e.second; };
    // auto lowerTriangularFilter = [](Edge e) { return e.first > e.second; };
    auto adj = pangolin::CSR<Index>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);
    LOG(debug, "nnz = {}", adj.nnz());
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "create CSR time {}s", elapsed);

    // read-mostly
    nvtxRangePush("read-mostly");
    const auto startHints = std::chrono::system_clock::now();
    if (opts.readMostly) {
      adj.read_mostly();
      for (const auto &gpu : gpus) {
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - startHints).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);

    // accessed-by
    start = std::chrono::system_clock::now();
    if (opts.accessedBy) {
      for (const auto &gpu : gpus) {
        adj.accessed_by(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "accessed-by CSR time {}s", elapsed);

    // prefetch
    start = std::chrono::system_clock::now();
    if (opts.prefetchAsync) {
      for (const auto &gpu : gpus) {
        adj.prefetch_async(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "prefetch CSR time {}s", elapsed);

    // create async counters
    start = std::chrono::system_clock::now();
    std::vector<pangolin::VertexDynTC> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(std::move(pangolin::VertexDynTC(dev)));
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "counter ctor time {}s", elapsed);

    // determine the number of edges per gpu
    const size_t rowsPerGPU = (adj.num_rows() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} rows per GPU", rowsPerGPU);

    // launch counting operations
    size_t rowStart = 0;
    for (auto &counter : counters) {
      const size_t rowStop = std::min(rowStart + rowsPerGPU, adj.num_rows());
      const size_t numRows = rowStop - rowStart;
      LOG(debug, "start async count on GPU {} ({} rows)", counter.device(), numRows);
      counter.count_async(adj.view(), rowStart, numRows, opts.blockSize);
      rowStart += rowsPerGPU;
    }

    // wait for counting operations to finish
    uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }

    elapsed = (std::chrono::system_clock::now() - startHints).count() / 1e9;
    LOG(info, "{} triangles ({} teps)", total, adj.nnz() / elapsed);
    LOG(info, "count time: {}s", elapsed);
    for (auto &counter : counters) {
      LOG(info, "GPU {} kernel time: {}", counter.device(), counter.kernel_time());
    }
    times.push_back(elapsed);
    if (counters.size() == 1) {
      kernelTimes.push_back(counters[0].kernel_time());
    } else {
      kernelTimes.push_back(0);
    }
    nodes = adj.num_rows();
    nnz = adj.nnz();
    tris = total;
  }

  if (opts.iters > 0) {
    fmt::print("dyn");
    fmt::print("{}{}", opts.sep, opts.blockSize);
    fmt::print("{}{}", opts.sep, opts.path);
    fmt::print("{}{}", opts.sep, nodes);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, tris);

    print_vec(times, opts.sep);
    for (const auto &s : times) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
    print_vec(kernelTimes, opts.sep);
    for (const auto &s : kernelTimes) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
  }

  fmt::print("\n");

  return 0;
}

int main(int argc, char **argv) {

  pangolin::init();

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool quiet = false;
  bool wide = false;
  bool header = false;

  RunOptions opts;

  opts.blockSize = 512;
  opts.iters = 1;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;
  opts.sep = ",";

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(quiet)["--quiet"]("only print errors");
  cli = cli | clara::Opt(wide)["--wide"]("64-bit node IDs");
  cli = cli | clara::Opt(header)["--header"]("only print CSV header");
  cli = cli | clara::Opt(opts.gpus, "dev ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(opts.blockSize, "N")["--bs"]("Number of threads in a block");
  cli = cli | clara::Opt(opts.readMostly)["--read-mostly"]("mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(opts.accessedBy)["--accessed-by"]("mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(opts.prefetchAsync)["--prefetch-async"]("prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(opts.iters, "N")["-n"]("number of counts");
  cli = cli | clara::Arg(opts.path, "graph file")("Path to adjacency list").required();

  auto result = cli.parse(clara::Args(argc, argv));
  if (!result) {
    LOG(critical, "Error in command line: {}", result.errorMessage());
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
  } else if (quiet) {
    pangolin::logger::set_level(pangolin::logger::Level::ERR);
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

  if (header) {
    print_header(opts);
  } else {
    if (wide) {
      LOG(debug, "64-bit node indices");
      return run<uint64_t>(opts);
    } else {
      LOG(debug, "32-bit node indices");
      return run<uint32_t>(opts);
    }
  }

  return 0;
}
