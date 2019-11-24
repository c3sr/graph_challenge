/*!

Count triangles using CUSparse

*/

#include <iostream>
#include <vector>

#include <nvToolsExt.h>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include "pangolin/configure.hpp"
#include "pangolin/file/tsv.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_val.hpp"
#include "pangolin/algorithm/csr/tc_cusparse.hpp"
#include "pangolin/cuda_cxx/stream.hpp"

struct RunOptions {
  int iters;
  std::vector<int> gpus;
  std::string path; //!< path for graph
  std::string sep;  //!< seperator for output

  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  bool shrinkToFit;
  bool preCountBarrier;
};

void print_header(const RunOptions &opts) {
  fmt::print("bmark{0}bs{0}gpus{0}graph{0}nodes{0}edges{0}tris", opts.sep);
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}total_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}total_teps{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}gpu_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}gpu_teps{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}count_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}count_teps{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}kernel_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}kernel_teps{}", opts.sep, i);
  }
  fmt::print("\n");
}

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

int run(RunOptions &opts) {

  // CUSparse uses integers for indices
  typedef int NodeIndex;
  typedef int EdgeIndex;
  typedef float Val;
  typedef pangolin::WeightedDiEdge<NodeIndex, Val> GraphEdge;
  typedef pangolin::CSR<NodeIndex, EdgeIndex, Val> CSR;
  typedef pangolin::file::TSV TSV;
  typedef TSV::edge_type FileEdge;
  typedef pangolin::CUSparseTC TC;
  using pangolin::Stream;

  auto gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // create a stream for each GPU
  std::vector<Stream> streams;
  for (const auto &gpu : gpus) {
    streams.push_back(Stream(gpu));
    LOG(debug, "created stream {} for gpu {}", streams.back(), gpu);
  }

  std::vector<double> totalTimes;
  std::vector<double> gpuTimes;
  std::vector<double> countTimes;
  std::vector<double> kernelTimes;
  uint64_t nnz;
  uint64_t numRows;
  uint64_t tris;
  // create csr and count `opts.iters` times
  for (int i = 0; i < opts.iters; ++i) {

    const auto totalStart = std::chrono::system_clock::now();

    // read data
    TSV file(opts.path);
    std::vector<FileEdge> fileEdges = file.read_edges();
    double elapsed = (std::chrono::system_clock::now() - totalStart).count() / 1e9;
    LOG(info, "read_data time {}s", elapsed);
    LOG(debug, "read {} edges", fileEdges.size());

    // build CSR
    CSR csr;
    for (auto fileEdge : fileEdges) {
      GraphEdge graphEdge;
      graphEdge.src = fileEdge.src;
      graphEdge.dst = fileEdge.dst;
      graphEdge.val = fileEdge.val;
      if (graphEdge.src > graphEdge.dst) {
        csr.add_next_edge(graphEdge);
      }
    }
    csr.finish_edges();

    if (opts.shrinkToFit) {
      LOG(debug, "shrink CSR");
      csr.shrink_to_fit();
    }

    elapsed = (std::chrono::system_clock::now() - totalStart).count() / 1e9;
    LOG(info, "io/csr time {}s", elapsed);
    LOG(debug, "CSR nnz = {} rows = {}", csr.nnz(), csr.num_rows());
    LOG(debug, "CSR cap = {}MB size = {}MB", csr.capacity_bytes() / 1024 / 1024, csr.size_bytes() / 1024 / 1024);

    const auto gpuStart = std::chrono::system_clock::now();

    // read-mostly
    nvtxRangePush("read-mostly");
    auto start = std::chrono::system_clock::now();
    if (opts.readMostly) {
      csr.read_mostly();
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);

    // accessed-by
    start = std::chrono::system_clock::now();
    if (opts.accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "accessed-by CSR time {}s", elapsed);

    // prefetch
    start = std::chrono::system_clock::now();
    if (opts.prefetchAsync) {
      for (size_t gpuIdx = 0; gpuIdx < gpus.size(); ++gpuIdx) {
        auto &gpu = gpus[gpuIdx];
        cudaStream_t stream = streams[gpuIdx].stream();
        csr.prefetch_async(gpu, stream);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "prefetch CSR time {}s", elapsed);

    if (opts.preCountBarrier) {
      LOG(debug, "sync streams after hints");
      for (auto &stream : streams) {
        stream.sync();
      }
    }

    // count triangles
    nvtxRangePush("count");
    const auto countStart = std::chrono::system_clock::now();
    TC counter(gpus[0]);
    uint64_t count = counter.count_sync(csr);


    const auto stop = std::chrono::system_clock::now();
    nvtxRangePop(); // count

    // record graph stats
    // tris = total;
    nnz = csr.nnz();
    numRows = csr.num_rows();

    const double totalElapsed = (stop - totalStart).count() / 1e9;
    const double gpuElapsed = (stop - gpuStart).count() / 1e9;
    const double countElapsed = (stop - countStart).count() / 1e9;
    LOG(info, "total time {}s ({} teps)", totalElapsed, nnz / totalElapsed);
    LOG(info, "gpu time   {}s ({} teps)", gpuElapsed, nnz / gpuElapsed);
    LOG(info, "count time {}s ({} teps)", countElapsed, nnz / countElapsed);
    totalTimes.push_back(totalElapsed);
    gpuTimes.push_back(gpuElapsed);
    countTimes.push_back(countElapsed);

    // for (auto &counter : counters) {
    //   double secs = counter.kernel_time();
    //   int dev = counter.device();
    //   LOG(info, "gpu {} kernel time {}s ({} teps)", dev, secs, nnz / secs);
    // }
    // if (counters.size() == 1) {
    //   kernelTimes.push_back(counters[0].kernel_time());
    // } else {
    //   kernelTimes.push_back(0);
    // }
  }

  if (opts.iters > 0) {
    fmt::print("binary");
    std::string gpuStr;
    for (auto gpu : gpus) {
      gpuStr += std::to_string(gpu);
    }
    fmt::print("{}{}", opts.sep, gpuStr);
    fmt::print("{}{}", opts.sep, opts.path);
    fmt::print("{}{}", opts.sep, numRows);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, tris);

    print_vec(totalTimes, opts.sep);
    for (const auto &s : totalTimes) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
    print_vec(gpuTimes, opts.sep);
    for (const auto &s : gpuTimes) {
      fmt::print("{}{}", opts.sep, nnz / s);
    }
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

int main(int argc, char **argv) {

  pangolin::init();

  RunOptions opts;
  opts.sep = ",";
  opts.iters = 1;
  opts.shrinkToFit = false;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;
  opts.preCountBarrier = true;

  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool onlyPrintHeader = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(onlyPrintHeader)["--header"]("print the header for the times output and quit");
  cli = cli | clara::Opt(opts.gpus, "dev ids")["-g"]("gpus to use");
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
  return run(opts);
}
