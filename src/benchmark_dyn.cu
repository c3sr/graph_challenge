/*!

Count triangles using warp-granularity dynamic algorithm selection

*/

#include <fmt/format.h>
#include <iostream>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

struct RunOptions {
  std::vector<int> gpus;
  std::string path;
  std::string sep;
  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  int dimBlock;
  int iters;
};

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

void print_header(RunOptions &opts) {
  fmt::print("dyn{0}bs{0}graph{0}nodes{0}edges{0}tris", opts.sep);

  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}time{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}teps{}", opts.sep, i);
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
  uint64_t nodes;
  uint64_t tris;
  uint64_t nnz;
  for (int i = 0; i < opts.iters; ++i) {
    // create csr
    start = std::chrono::system_clock::now();
    auto upperTriangularFilter = [](Edge e) { return e.first < e.second; };
    auto lowerTriangularFilter = [](Edge e) { return e.first > e.second; };
    auto adj = pangolin::CSRCOO<Index>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);
    LOG(debug, "nnz = {}", adj.nnz());
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "create CSR time {}s", elapsed);

    fmt::print("dyn");
    fmt::print("{}{}", opts.sep, opts.dimBlock);
    fmt::print("{}{}", opts.sep, opts.path);

    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (opts.readMostly) {
      adj.read_mostly();
      for (const auto &gpu : gpus) {
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
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

    // count triangles
    start = std::chrono::system_clock::now();

    // create async counters
    std::vector<pangolin::EdgeWarpDynTC> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(std::move(pangolin::EdgeWarpDynTC(dev)));
    }

    // determine the number of edges per gpu
    const size_t edgesPerGPU = (adj.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    size_t edgeStart = 0;
    for (auto &counter : counters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, adj.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(), numEdges);
      counter.count_async(adj.view(), edgeStart, numEdges, opts.dimBlock);
      edgeStart += edgesPerGPU;
    }

    // wait for counting operations to finish
    uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "count time {}s", elapsed);
    LOG(info, "{} triangles ({} teps)", total, adj.nnz() / elapsed);
    for (auto &counter : counters) {
      LOG(debug, "GPU {} kernel time: {}", counter.device(), counter.kernel_time());
    }
    times.push_back(elapsed);
    nodes = adj.num_rows();
    nnz = adj.nnz();
    tris = total;
  }

  if (opts.iters > 0) {
    fmt::print("{}{}", opts.sep, nodes);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, tris);

    print_vec(times, opts.sep);
    for (const auto &s : times) {
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
  bool wide = false;
  bool header = false;

  RunOptions opts;

  opts.dimBlock = 512;
  opts.iters = 1;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;
  opts.sep = ",";

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(wide)["--wide"]("64-bit node IDs");
  cli = cli | clara::Opt(header)["--header"]("only print CSV header");
  cli = cli | clara::Opt(opts.gpus, "dev ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(opts.dimBlock, "block-dim")["--bs"]("Number of threads in a block");
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
