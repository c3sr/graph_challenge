#include <fmt/format.h>
#include <iostream>

#include <cmath>
#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {
  std::string path;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
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
  // create csr
  start = std::chrono::system_clock::now();
  auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first < e.second;
  };
  auto csr = pangolin::COO<uint64_t>::from_edges(edges.begin(), edges.end(),
                                                 upperTriangular);
  LOG(debug, "nnz = {}", csr.nnz());
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "create CSR time {}s", elapsed);

  if (0 == csr.num_rows()) {
    LOG(warn, "empty CSR");
    return 0;
  }

  std::vector<std::vector<uint64_t>> bins(64);
  for (auto &bin : bins) {
    bin = std::vector<uint64_t>(64, 0);
  }

  // count triangles
  nvtxRangePush("histogram");
  start = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < csr.nnz(); ++i) {
    uint64_t src = csr.row_ind()[i];
    uint64_t dst = csr.col_ind()[i];

    double srcLen = csr.row_ptr()[src + 1] - csr.row_ptr()[src];
    double dstLen = csr.row_ptr()[dst + 1] - csr.row_ptr()[dst];

    size_t srcBin = 0;
    if (srcLen != 0) {
      srcBin = std::ceil(std::log2(srcLen)) + 1;
    }
    size_t dstBin = 0;
    if (dstLen != 0) {
      dstBin = std::ceil(std::log2(dstLen)) + 1;
    }

    std::cout << fmt::format("{}->{}, {}->{}", srcLen, srcBin, dstLen, dstBin)
              << std::endl;

#pragma omp atomic
    bins[srcBin][dstBin]++;
  }

  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  nvtxRangePop();
  LOG(info, "histogram time {}s", elapsed);

  // find the maximum bin
  size_t iMax = 0;
  for (size_t i = 0; i < bins.size(); ++i) {
    for (size_t j = 0; j < bins[i].size(); ++j) {
      if (bins[i][j] > 0) {
        iMax = i > iMax ? i : iMax;
        iMax = j > iMax ? j : iMax;
      }
    }
  }
  iMax++;

  for (size_t i = 0; i < iMax; ++i) {
    std::cout << ",\t" << i;
  }
  std::cout << std::endl;
  for (size_t i = 0; i < iMax; ++i) {
    const auto &row = bins[i];
    std::cout << i << "\t";
    for (size_t j = 0; j < iMax; ++j) {
      std::cout << row[j] << ",\t";
    }
    std::cout << std::endl;
  }

  return 0;
}
