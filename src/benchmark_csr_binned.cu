/*!

Count triangles using warp-granularity dynamic algorithm selection

*/

#include <iostream>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include <fmt/format.h>

#include "pangolin/algorithm/tc_vertex_cpu.cuh"
#include "pangolin/algorithm/tc_vertex_dyn.cuh"
#include "pangolin/bounded_buffer.hpp"
#include "pangolin/configure.hpp"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_binned.hpp"

using pangolin::BoundedBuffer;
// Buffer is a BoundedBuffer with two entries (double buffer)
template <typename T> using Buffer = BoundedBuffer<T, 2>;

/*! Producer reads edges from a file and fills a buffer
 */
template <typename Edge> void produce(const std::string path, Buffer<std::vector<Edge>> &queue) {
  double readTime = 0, queueTime = 0;
  pangolin::EdgeListFile file(path);

  std::vector<Edge> edges;

  while (true) {
    auto readStart = std::chrono::system_clock::now();
    size_t readCount = file.get_edges(edges, 500);
    auto readEnd = std::chrono::system_clock::now();
    readTime += (readEnd - readStart).count() / 1e9;
    SPDLOG_TRACE(pangolin::logger::console(), "reader: read {} edges", edges.size());
    if (0 == readCount) {
      break;
    }

    auto queueStart = std::chrono::system_clock::now();
    queue.push(std::move(edges));
    auto queueEnd = std::chrono::system_clock::now();
    queueTime += (queueEnd - queueStart).count() / 1e9;
    SPDLOG_TRACE(pangolin::logger::console(), "reader: pushed edges");
  }

  SPDLOG_TRACE(pangolin::logger::console(), "reader: closing queue");
  queue.close();
  LOG(debug, "reader: {}s I/O, {}s blocked", readTime, queueTime);
}

/*! consumer reads edges from a queue and adds the mto a matrix
 */
template <typename Mat> void consume(Buffer<std::vector<typename Mat::edge_type>> &queue, Mat &mat) {
  typedef typename Mat::edge_index_type EdgeIndex;
  typedef typename Mat::node_index_type NodeIndex;
  typedef typename Mat::edge_type Edge;

  double queueTime = 0, csrTime = 0;
  auto upperTriangular = [](const Edge &e) { return e.first < e.second; };

  // keep grabbing while queue is filling
  NodeIndex maxNode = 0;
  while (true) {
    std::vector<Edge> edges;
    bool popped;
    SPDLOG_TRACE(pangolin::logger::console(), "builder: trying to pop...");
    auto queueStart = std::chrono::system_clock::now();
    edges = queue.pop(popped);
    auto queueEnd = std::chrono::system_clock::now();
    queueTime += (queueEnd - queueStart).count() / 1e9;
    if (popped) {
      SPDLOG_TRACE(pangolin::logger::console(), "builder: popped {} edges", edges.size());
      auto csrStart = std::chrono::system_clock::now();
      for (const auto &edge : edges) {
        maxNode = max(edge.first, maxNode);
        maxNode = max(edge.second, maxNode);
        if (upperTriangular(edge)) {
          // SPDLOG_TRACE(pangolin::logger::console(), "{} {}", edge.first, edge.second);
          mat.add_next_edge(edge);
        }
      }
      auto csrEnd = std::chrono::system_clock::now();
      csrTime += (csrEnd - csrStart).count() / 1e9;
    } else {
      SPDLOG_TRACE(pangolin::logger::console(), "builder: no edges after pop");
      assert(queue.empty());
      assert(queue.closed());
      break;
    }
  }

  auto csrStart = std::chrono::system_clock::now();
  mat.finish_edges(maxNode);
  auto csrEnd = std::chrono::system_clock::now();
  csrTime += (csrEnd - csrStart).count() / 1e9;

  LOG(debug, "builder: {}s csr {}s blocked", csrTime, queueTime);
}

struct RunOptions {
  std::vector<int> gpus;
  std::string path;
  std::string sep;
  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  int blockSize;
  int iters;
  uint64_t maxExpectedNode;
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

template <typename NodeIndex, typename EdgeIndex> int run(RunOptions &opts) {
  typedef pangolin::CSRBinned<NodeIndex, EdgeIndex> CSR;
  typedef typename CSR::edge_type Edge;
  typedef typename pangolin::VertexDynTC TC;
  // typedef typename pangolin::VertexCPUTC TC;

  std::vector<int> gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // read data / build
  auto start = std::chrono::system_clock::now();

  Buffer<std::vector<Edge>> queue;
  CSR adj(opts.maxExpectedNode);

  uint64_t expectedNnz = opts.maxExpectedNode * 20;
  LOG(debug, "reserve {} rows {} nnz", opts.maxExpectedNode, expectedNnz);
  adj.reserve(opts.maxExpectedNode, expectedNnz);

  // start a thread to read the matrix data
  LOG(debug, "start disk reader");
  std::thread reader(produce<Edge>, opts.path, std::ref(queue));

  // start a thread to build the matrix
  LOG(debug, "start csr build");
  std::thread builder(consume<CSR>, std::ref(queue), std::ref(adj));
  // consume(queue, csr, &readerActive);

  LOG(debug, "waiting for disk reader...");
  reader.join();
  LOG(debug, "waiting for CSR builder...");
  builder.join();
  assert(queue.empty());

  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data/build time {}s", elapsed);
  LOG(debug, "CSR: nnz = {}, rows = {}", adj.nnz(), adj.num_rows());

  // EdgeIndex computedNnz = 0;
  // for (size_t i = 0; i < adj.num_partitions(); ++i) {
  //   computedNnz += adj.view(i).part_nnz();
  // }
  // LOG(debug, "CSR: nnz by parts = {}", computedNnz);
  // LOG(debug, "CSR: nnz by parts = {}", adj.view(0, 4).part_nnz() + adj.view(4, 8).part_nnz());

  // create csr and count `iters` times
  std::vector<double> times;
  std::vector<double> kernelTimes;
  uint64_t nodes;
  uint64_t tris;
  uint64_t nnz;
  for (int i = 0; i < opts.iters; ++i) {
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
    std::vector<TC> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(std::move(TC(dev)));
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "counter ctor time {}s", elapsed);

    // assign a different partition to each gpu

    // determine the number of partitions per gpu
    // {
    //   uint64_t total = 0;
    //   for (size_t task = 0; task < adj.num_partitions(); ++task) {
    //     auto view = adj.view(task);
    //     uint64_t count = counters[0].count_sync(view, opts.blockSize);
    //     LOG(debug, "part {} count {}", task, count);
    //     total += count;
    //   }
    //   LOG(debug, "TOTAL {}", total);
    // }

    // determine the number of partitions per gpu
    size_t partsLeft = adj.num_partitions();
    for (size_t task = 0; task < counters.size(); ++task) {
      size_t taskParts;
      if (task + 1 < counters.size()) {
        taskParts = (double(partsLeft) / std::sqrt(counters.size()) + 0.5); // round to nearest int
      } else {
        taskParts = partsLeft;
      }

      size_t partStart = adj.num_partitions() - partsLeft;
      auto view = adj.view(partStart, partStart + taskParts);

      LOG(debug, "start async count on GPU {} (partition {}-{})", counters[task].device(), partStart,
          partStart + taskParts);
      counters[task].count_async(view, opts.blockSize);

      partsLeft -= taskParts;
    }

    /*
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
    */

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
  opts.maxExpectedNode = 0;

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
  cli = cli | clara::Arg(opts.maxExpectedNode, "UINT")("Max node ID").required();
  cli = cli | clara::Arg(opts.path, "STR")("Path to adjacency list").required();

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

  if (opts.maxExpectedNode == 0) {
    LOG(critical, "must provide non-zero maxExpectedNode");
    exit(1);
  }

  if (header) {
    print_header(opts);
  } else {
    if (wide) {
      LOG(debug, "64-bit edge indices");
      return run<uint32_t, uint64_t>(opts);
    } else {
      LOG(debug, "32-bit edge indices");
      return run<uint32_t, uint32_t>(opts);
    }
  }

  return 0;
}
