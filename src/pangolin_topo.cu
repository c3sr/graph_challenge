/*! Display pangolin's detected system topology

*/

#include <fmt/format.h>
#include <iostream>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {

  pangolin::init();

  bool help = false;
  bool debug = false;
  bool verbose = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");

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

  pangolin::topology::Topology topology = pangolin::topology::topology();

  // summarize NUMA regions
  fmt::print(stdout, "NUMA regions\n");
  if (topology.numas_.empty()) {
    fmt::print(stdout, "  no NUMA regions detected");
  } else {
    for (const auto numaKv : topology.numas_) {
      auto numa = numaKv.second;
      fmt::print(stdout, "  numa {}\n", numa->id_);
      for (const auto cpu : numa->cpus_) {
        fmt::print(stdout, "    cpu {}\n", cpu->id_);
      }
      for (const auto gpu : numa->gpus_) {
        fmt::print(stdout, "    gpu {}\n", gpu->cudaId_);
      }
    }
  }

  // summarize CPU->GPU affinity
  fmt::print(stdout, "CPU->GPU affinity\n");
  for (const auto cpuKv : topology.cpus_) {
    auto cpu = cpuKv.second;
    fmt::print(stdout, "  cpu {}\n", cpu->id_);
    for (const auto gpu : cpu->gpus_) {
      fmt::print(stdout, "    gpu {}\n", gpu->cudaId_);
    }
  }

  fmt::print(stdout, "GPU->CPU affinity\n");
  for (const auto gpuKv : topology.cudaGpus_) {
    auto gpu = gpuKv.second;
    fmt::print(stdout, "  gpu {}\n", gpu->cudaId_);
    for (const auto cpu : gpu->cpus_) {
      fmt::print(stdout, "    cpu {}\n", cpu->id_);
    }
  }

  return 0;
}
