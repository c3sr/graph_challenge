#include <fmt/format.h>
#include <iostream>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {

  pangolin::Config config;

  std::vector<int> gpus;
  std::string path;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(config.numCPUThreads_,
                         "int")["-c"]["--num_cpu"]("number of cpu threads");
  cli = cli | clara::Opt(gpus, "ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(config.hints_)["--unified-memory-hints"](
                  "use unified memory hints");
  cli = cli | clara::Opt(config.storage_, "zc|um")["-s"]("GPU memory kind");
  cli = cli |
        clara::Opt(
            config.type_,
            "cpu|csr|cudamemcpy|edge|hu|impact2018|impact2018|nvgraph|vertex")
            ["-m"]["--method"]("method")
                .required();
  cli = cli | clara::Opt(config.kernel_, "string")["-k"]["--kernel"]("kernel");
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

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  std::vector<pangolin::EdgeTy<uint64_t>> edges;
  std::vector<pangolin::EdgeTy<uint64_t>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr
  start = std::chrono::system_clock::now();
  auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first < e.second;
  };
  auto csr = pangolin::COO<uint64_t>::from_edges(edges.begin(), edges.end(),
                                                 upperTriangular);
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "create CSR time {}s", elapsed);

  // count triangles
  start = std::chrono::system_clock::now();

  // create async counters
  std::vector<pangolin::LinearTC> counters;
  for (int dev : gpus) {
    counters.push_back(pangolin::LinearTC(dev));
  }

  // determine the number of edges per gpu
  size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();

  // launch counting operations
  size_t edgeStart = 0;
  for (auto &counter : counters) {
    const size_t edgeStop = std::max(edgeStart + edgesPerGPU, csr.nnz());
    const size_t numEdges = edgeStop - edgeStart;
    counter.count_async(csr, edgeStart, numEdges);
    edgeStart += edgesPerGPU;
  }

  // wait for counting operations to finish
  uint64_t total = 0;
  for (auto &counter : counters) {
    counter.sync();
    total += counter.count();
  }

  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "count time {}s", elapsed);
  LOG(info, "{} triangles", total);

  return 0;
}
