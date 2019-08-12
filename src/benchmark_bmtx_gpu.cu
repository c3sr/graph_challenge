/*!

Count triangles using the per-edge binary search

*/

#include <iostream>

#include <nvToolsExt.h>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include "pangolin/algorithm/tc_task_gpu.cuh"
#include "pangolin/configure.hpp"
#include "pangolin/file/bmtx_stream.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_binned.hpp"

struct RunOptions {
  std::string path; //!< path for graph
  std::string sep;  //!< seperator for output
  std::vector<int> gpus;
  int dimBlock;
  int iters;

  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
};

template <typename NodeIndex, typename EdgeIndex> int run(RunOptions &opts) {

  using namespace pangolin;

  typedef typename pangolin::EdgeTy<NodeIndex> Edge;
  typedef TaskGPUTC TC;
  typedef TC::Task Task;

  auto gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // make streams
  std::vector<Stream> streams;
  for (const auto gpu : gpus) {
    streams.push_back(Stream(gpu));
  }

  // read data
  auto start = std::chrono::system_clock::now();
  auto bmtx = pangolin::open_bmtx_stream<NodeIndex>(opts.path);
  LOG(info, "{}: rows={} cols={} entries={}", opts.path, bmtx.num_rows(), bmtx.num_cols(), bmtx.nnz());

  std::vector<Edge> edges;
  {
    Edge edge;
    while (bmtx.readEdge(edge)) {
      edges.push_back(edge);
    }
  }

  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  start = std::chrono::system_clock::now();
  CSRBinned<NodeIndex, EdgeIndex> csr(bmtx.num_rows(), bmtx.nnz());

  for (const auto &edge : edges) {
    csr.add_next_edge(edge);
  }
  csr.finish_edges();

  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "csr time {}s", elapsed);

  // Make counters
  std::vector<TC> counters;
  for (size_t i = 0; i < gpus.size(); ++i) {
    counters.push_back(std::move(TC(streams[i].ref())));
  }

  // Build tasks
  std::vector<Task> tasks;
  for (size_t i = 0; i < csr.num_partitions(); ++i) {
    for (size_t j = 0; j < csr.num_partitions(); ++j) {
      for (size_t k = 0; k < csr.num_partitions(); ++k) {
        tasks.push_back({i, j, k});
      }
    }
  }
  LOG(info, "{} tasks", tasks.size());

  size_t gpuIdx = 0;
  for (const auto task : tasks) {
    auto &tc = counters[gpuIdx];
    LOG(debug, "task {} {} {} on counter {}", task.i, task.j, task.k, gpuIdx);
    tc.count_async(csr.two_col_view(task.j, task.k), task);
    gpuIdx = (gpuIdx + 1) % gpus.size();
  }

  uint64_t total = 0;
  for (auto &counter : counters) {
    LOG(info, "waiting on GPU {}", counter.device());
    counter.sync();
    total += counter.count();
  }
  fmt::print("{}\n", total);

  return 0;
}

void print_header(const RunOptions &opts) {
  fmt::print("benchmark{0}bs{0}graph{0}nodes{0}edges{0}tris", opts.sep);
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
    fmt::print("{}count{}", opts.sep, i);
  }
  for (int i = 0; i < opts.iters; ++i) {
    fmt::print("{}count_teps{}", opts.sep, i);
  }
  fmt::print("\n");
}

int main(int argc, char **argv) {

  pangolin::init();

  RunOptions opts;
  opts.sep = ",";
  opts.dimBlock = 512;
  opts.iters = 1;
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

  if (onlyPrintHeader) {
    print_header(opts);
    return 0;
  }
  if (wide) {
    return run<uint32_t, uint64_t>(opts);
  } else {
    return run<uint32_t, uint32_t>(opts);
  }
}
