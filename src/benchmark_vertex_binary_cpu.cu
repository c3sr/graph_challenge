/*!

Count triangles using CPUVertexBinaryTC

*/

#include <iostream>
#include <vector>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include "pangolin/algorithm/csr/tc_vertex_binary_cpu.hpp"
#include "pangolin/configure.hpp"
#include "pangolin/file/tsv.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr.hpp"

struct RunOptions {
  int iters;
  std::string path; //!< path for graph
  std::string sep;  //!< seperator for output

  int numThreads;
  bool shrinkToFit;
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

template <typename Index> int run(RunOptions &opts) {

  typedef Index NodeIndex;
  typedef Index EdgeIndex;
  typedef pangolin::EdgeTy<NodeIndex> GraphEdge;
  typedef pangolin::CSR<Index> CSR;
  typedef pangolin::file::TSV TSV;
  typedef TSV::edge_type FileEdge;
  typedef pangolin::CPUVertexBinaryTC TC;

  std::vector<double> totalTimes;
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
      graphEdge.first = fileEdge.src;
      graphEdge.second = fileEdge.dst;
      if (graphEdge.first > graphEdge.second) {
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

    // count triangles
    const auto countStart = std::chrono::system_clock::now();
    TC counter(opts.numThreads);
    tris = counter.count_sync(csr);
    const auto stop = std::chrono::system_clock::now();

    // record graph stats
    nnz = csr.nnz();
    numRows = csr.num_rows();

    const double totalElapsed = (stop - totalStart).count() / 1e9;
    const double countElapsed = (stop - countStart).count() / 1e9;
    LOG(info, "total time {}s ({} teps)", totalElapsed, nnz / totalElapsed);
    LOG(info, "count time {}s ({} teps)", countElapsed, nnz / countElapsed);
    totalTimes.push_back(totalElapsed);
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
    fmt::print("nvgraph");
    fmt::print("{}{}", opts.sep, opts.path);
    fmt::print("{}{}", opts.sep, numRows);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, tris);

    print_vec(totalTimes, opts.sep);
    for (const auto &s : totalTimes) {
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
  opts.numThreads = 1;

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
  cli = cli | clara::Opt(opts.shrinkToFit)["--shrink-to-fit"]("shrink allocations to fit data");
  cli = cli | clara::Opt(opts.iters, "N")["-n"]("number of counts");
  cli = cli | clara::Opt(opts.numThreads, "INT")["-t"]("number of threads");
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
