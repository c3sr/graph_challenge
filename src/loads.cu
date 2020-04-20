/*! \file Compute the number of loads needed during a triangle count
 */

#include <fmt/format.h>
#include <iostream>

#include <cmath>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

constexpr size_t SCALE_B = 1;
constexpr size_t SCALE_K = 1024;
constexpr size_t SCALE_M = 1024 * 1024;
constexpr size_t SCALE_G = 1024ull * 1024ull * 1024ull;
const char *UNIT_B = "B";
const char *UNIT_K = "KB";
const char *UNIT_M = "MB";
const char *UNIT_G = "GB";

int main(int argc, char **argv) {
  std::string path;
  bool help = false;
  bool debug = false;
  bool verbose = false;
  size_t dataSize = 8;
  std::string unitStr = "M";

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(dataSize, "BYTES")["-d"]("size of each data load in bytes (8)");
  cli = cli | clara::Opt(unitStr, "BKMG")["-u"]("the unit to present results in (M)");
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

  // set logging level
  if (verbose) {
    pangolin::logger::set_level(pangolin::logger::Level::TRACE);
  } else if (debug) {
    pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  }

  LOG(debug, "pangolin version: {}.{}.{}", PANGOLIN_VERSION_MAJOR, PANGOLIN_VERSION_MINOR, PANGOLIN_VERSION_PATCH);
  LOG(debug, "pangolin branch:  {}", PANGOLIN_GIT_REFSPEC);
  LOG(debug, "pangolin sha:     {}", PANGOLIN_GIT_HASH);
  LOG(debug, "pangolin changes: {}", PANGOLIN_GIT_LOCAL_CHANGES);

#ifndef NDEBUG
  LOG(warn, "Not a release build");
#endif

  const char *unit = UNIT_M;
  size_t scale = SCALE_M;
  if (unitStr == "B") {
    unit = UNIT_B;
    scale = SCALE_B;
  } else if ("K" == unitStr) {
    unit = UNIT_K;
    scale = SCALE_K;
  } else if ("M" == unitStr) {
    unit = UNIT_M;
    scale = SCALE_M;
  } else if ("G" == unitStr) {
    unit = UNIT_G;
    scale = SCALE_G;
  } else {
    std::cout << cli;
    return -1;
  }

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  std::vector<pangolin::DiEdge<uint64_t>> edges;
  std::vector<pangolin::DiEdge<uint64_t>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr
  start = std::chrono::system_clock::now();
  auto upperTriangular = [](pangolin::DiEdge<uint64_t> e) { return e.src < e.dst; };
  auto csr = pangolin::CSRCOO<uint64_t>::from_edges(edges.begin(), edges.end(), upperTriangular);
  LOG(debug, "nnz = {}", csr.nnz());
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "create CSR time {}s", elapsed);

  if (0 == csr.num_rows()) {
    LOG(warn, "empty CSR");
    return 0;
  }

  uint64_t rowIndBytes = 0;
  uint64_t colIndBytes = 0;
  uint64_t rowPtrBytes = 0;

  // count loads
  start = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic) reduction(+ : colIndBytes, rowIndBytes, rowPtrBytes)
  for (size_t i = 0; i < csr.nnz(); ++i) {
    uint64_t src = csr.row_ind()[i];
    uint64_t dst = csr.col_ind()[i];
    rowIndBytes += dataSize;
    colIndBytes += dataSize;

    size_t srcLen = csr.row_ptr()[src + 1] - csr.row_ptr()[src];
    size_t dstLen = csr.row_ptr()[dst + 1] - csr.row_ptr()[dst];
    rowPtrBytes += 4 * dataSize;

    colIndBytes += (srcLen + dstLen) * dataSize;
  }

  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "load count time {}s", elapsed);

  const uint64_t staticMatBytes = (csr.nnz() * 2 * 8 + (csr.num_rows() + 1) * 8);
  const uint64_t dynamicMatBytes = rowPtrBytes + rowIndBytes + colIndBytes;

  std::cout << "static(" << unit << "),\t"
            << "rowPtr(" << unit << "),\t"
            << "rowInd(" << unit << "),\t"
            << "colInd(" << unit << "),\t"
            << "dynamic(" << unit << "),\t"
            << "scale" << std::endl;
  std::cout << staticMatBytes / double(scale) << ",\t" << rowPtrBytes / double(scale) << ",\t"
            << rowIndBytes / double(scale) << ",\t" << colIndBytes / double(scale) << ",\t"
            << dynamicMatBytes / double(scale) << ",\t" << dynamicMatBytes / double(staticMatBytes) << std::endl;

  return 0;
}
