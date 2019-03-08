#include <fmt/format.h>
#include <iostream>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {

  std::vector<int> gpus;
  std::string path;
  int coarsening = 1;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(gpus, "ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(coarsening,
                         "coarsening")["-c"]("Number of elements per thread");
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
  auto upperTriangularFilter = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first < e.second;
  };
  auto lowerTriangularFilter = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first > e.second;
  };
  auto csr = pangolin::COO<uint64_t>::from_edges(edges.begin(), edges.end(),
                                                 upperTriangularFilter);
  LOG(debug, "nnz = {}", csr.nnz());
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "create CSR time {}s", elapsed);

  // count triangles
  start = std::chrono::system_clock::now();

  // create async counters
  std::vector<pangolin::BinaryTC> counters;
  for (int dev : gpus) {
    LOG(debug, "create device {} counter", dev);
    counters.push_back(pangolin::BinaryTC(dev));
  }

  // determine the number of edges per gpu
  const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
  LOG(debug, "{} edges per GPU", edgesPerGPU);

  // launch counting operations
  size_t edgeStart = 0;
  for (auto &counter : counters) {
    const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
    const size_t numEdges = edgeStop - edgeStart;
    LOG(debug, "start async count on GPU {}", counter.device());
    counter.count_async(csr.view(), numEdges, edgeStart, coarsening);
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
  LOG(info, "{} triangles ({} teps)", total, csr.nnz() / elapsed);

  return 0;
}
