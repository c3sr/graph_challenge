/*!

Compare CSR construction when overlapped with io or not

*/

#include <iostream>

#include <thread>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include <nvToolsExt.h>

#include "pangolin/bounded_buffer.hpp"
#include "pangolin/configure.hpp"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr.hpp"
#include "pangolin/sparse/csr_coo.hpp"

using pangolin::BoundedBuffer;

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

struct WorkerTimes {
  double wait; //<! time waiting for synchronization
  double work; //<! time doing something useful (io or csr)
};

// Buffer is a BoundedBuffer with two entries (double buffer)
template <typename T> using Buffer = BoundedBuffer<T, 2>;

template <typename Edge> void produce(const std::string path, Buffer<std::vector<Edge>> &queue, WorkerTimes &times) {
  times.wait = 0;
  times.work = 0;

  pangolin::EdgeListFile file(path);
  std::vector<Edge> edges;

  while (true) {
    auto readStart = std::chrono::system_clock::now();
    size_t readCount = file.get_edges(edges, 1000);
    auto readEnd = std::chrono::system_clock::now();
    times.work += (readEnd - readStart).count() / 1e9;
    SPDLOG_TRACE(pangolin::logger::console(), "reader: read {} edges", edges.size());
    if (0 == readCount) {
      break;
    }

    auto queueStart = std::chrono::system_clock::now();
    queue.push(std::move(edges));
    auto queueEnd = std::chrono::system_clock::now();
    times.wait += (queueEnd - queueStart).count() / 1e9;
    SPDLOG_TRACE(pangolin::logger::console(), "reader: pushed edges");
  }

  SPDLOG_TRACE(pangolin::logger::console(), "reader: closing queue");
  queue.close();
  LOG(debug, "reader: {}s I/O, {}s blocked", times.work, times.wait);
}

template <typename Mat>
void consume(Buffer<std::vector<typename Mat::edge_type>> &queue, Mat &mat, WorkerTimes &times) {
  typedef typename Mat::index_type Index;
  typedef typename Mat::edge_type Edge;

  auto upperTriangularFilter = [](Edge e) { return e.first < e.second; };
  // auto lowerTriangularFilter = [](Edge e) { return e.first > e.second; };

  times.wait = 0;
  times.work = 0;

  // keep grabbing while queue is filling
  Index maxNode = 0;
  while (true) {
    std::vector<Edge> edges;
    bool popped;
    SPDLOG_TRACE(pangolin::logger::console(), "builder: trying to pop...");
    auto queueStart = std::chrono::system_clock::now();
    edges = queue.pop(popped);
    auto queueEnd = std::chrono::system_clock::now();
    times.wait += (queueEnd - queueStart).count() / 1e9;
    if (popped) {
      SPDLOG_TRACE(pangolin::logger::console(), "builder: popped {} edges", edges.size());
      auto csrStart = std::chrono::system_clock::now();
      for (const auto &edge : edges) {
        maxNode = max(edge.first, maxNode);
        maxNode = max(edge.second, maxNode);
        if (upperTriangularFilter(edge)) {
          // SPDLOG_TRACE(pangolin::logger::console(), "{} {}", edge.first, edge.second);
          mat.add_next_edge(edge);
        }
      }
      auto csrEnd = std::chrono::system_clock::now();
      times.work += (csrEnd - csrStart).count() / 1e9;
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
  times.work += (csrEnd - csrStart).count() / 1e9;

  LOG(debug, "builder: {}s csr {}s blocked", times.work, times.wait);
}

enum class Type { SEQUENTIAL, OVERLAPPED};

struct RunOptions {
  int iters;
  std::vector<int> gpus;
  std::string path;
  std::string sep;

  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;

  Type type;
};

void print_header(const RunOptions &opts) {
  fmt::print("bmark{0}gpus{0}graph{0}nodes{0}edges", opts.sep);
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}total_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}io_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}readmostly_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}accessedby_time{}", opts.sep, i);
  }
  for (auto i = 0; i < opts.iters; ++i) {
    fmt::print("{}prefetch_time{}", opts.sep, i);
  }
  fmt::print("\n");
}

template <typename Index> void build_overlapped(pangolin::CSRCOO<Index> &csr, double &ioTime, RunOptions &opts) {
  typedef pangolin::EdgeTy<Index> Edge;

  Buffer<std::vector<Edge>> queue;
  WorkerTimes builderTimes, consumerTimes;
  // start a thread to read the matrix data
  LOG(debug, "start disk reader");
  std::thread reader(produce<Edge>, opts.path, std::ref(queue), std::ref(consumerTimes));

  // start a thread to build the matrix
  LOG(debug, "start csr build");
  std::thread builder(consume<pangolin::CSRCOO<Index>>, std::ref(queue), std::ref(csr), std::ref(builderTimes));
  // consume(queue, csr, &readerActive);

  LOG(debug, "waiting for disk reader...");
  reader.join();
  LOG(debug, "waiting for CSR builder...");
  builder.join();
  assert(queue.empty());

  ioTime = 0; // no separate io measurable
}

template <typename Index> void build_sequential(pangolin::CSRCOO<Index> &csr, double &ioTime, RunOptions &opts) {
  typedef pangolin::EdgeTy<Index> Edge;
  auto upperTriangularFilter = [](Edge e) { return e.first < e.second; };
  // auto lowerTriangularFilter = [](Edge e) { return e.first > e.second; };

  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(opts.path);

  std::vector<Edge> edges;
  std::vector<Edge> fileEdges;
  while (file.get_edges(fileEdges, 500)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  ioTime = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(debug, "sequential IO time was {}", ioTime);

  csr = pangolin::CSRCOO<Index>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);
}

template <typename Index> int run(RunOptions &opts) {
  typedef pangolin::EdgeTy<Index> Edge;
  using pangolin::RcStream;

  // times to report
  std::vector<double> totalTimes;
  std::vector<double> ioTimes;
  std::vector<double> readMostlyTimes;
  std::vector<double> accessedByTimes;
  std::vector<double> prefetchAsyncTimes;

  // other things to report
  uint64_t numRows;
  uint64_t nnz;

  std::vector<int> gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // create a stream for each GPU
  std::vector<RcStream> streams;
  for (const auto &gpu : gpus) {
    streams.push_back(RcStream(gpu));
  }

  for (int i = 0; i < opts.iters; ++i) {

    auto iterStart = std::chrono::system_clock::now();

    // the time spent on disk, if measureable
    double ioTime = 0;

    // build the csr
    pangolin::CSRCOO<Index> csr;
    if (opts.type == Type::OVERLAPPED) {
      build_overlapped(csr, ioTime, opts);
    } else if (opts.type == Type::SEQUENTIAL) {
      build_sequential(csr, ioTime, opts);
    } else {
      LOG(critical, "unexpected type");
      exit(1);
    }

    nnz = csr.nnz();
    numRows = csr.num_rows();
    LOG(debug, "CSR: nnz = {}, rows = {}", csr.nnz(), csr.num_rows());

    ioTimes.push_back(ioTime);

    // read-mostly
    auto start = std::chrono::system_clock::now();
    if (opts.readMostly) {
      csr.read_mostly();
    }
    double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "read-mostly CSR time {}s", elapsed);
    readMostlyTimes.push_back(elapsed);

    // accessed-by
    start = std::chrono::system_clock::now();
    if (opts.accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "accessed-by CSR time {}s", elapsed);
    accessedByTimes.push_back(elapsed);

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
    prefetchAsyncTimes.push_back(elapsed);

    if (opts.readMostly || opts.accessedBy || opts.prefetchAsync) {
      LOG(debug, "sync streams after hints");
      for (auto stream : streams) {
        stream.sync();
      }
    }

    elapsed = (std::chrono::system_clock::now() - iterStart).count() / 1e9;
    totalTimes.push_back(elapsed);
  }

  if (opts.iters > 0) {
    switch (opts.type) {
    case Type::OVERLAPPED: {
      fmt::print("overlap");
      break;
    }
    case Type::SEQUENTIAL: {
      fmt::print("sequential");
      break;
    }
    default: {
      LOG(critical, "type not provided.");
      exit(1);
    }
    }
    std::string gpuStr;
    for (auto gpu : gpus) {
      gpuStr += std::to_string(gpu);
    }
    fmt::print("{}{}", opts.sep, gpuStr);
    fmt::print("{}{}", opts.sep, opts.path);
    fmt::print("{}{}", opts.sep, nnz);
    fmt::print("{}{}", opts.sep, numRows);

    print_vec(totalTimes, opts.sep);
    print_vec(ioTimes, opts.sep);
    print_vec(readMostlyTimes, opts.sep);
    print_vec(accessedByTimes, opts.sep);
    print_vec(prefetchAsyncTimes, opts.sep);
  }
  fmt::print("\n");

  return 0;
}

int main(int argc, char **argv) {

  pangolin::init();

  // options not passed to run
  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool wide = false;
  bool header = false;
  bool overlap = false;

  // options passed to run
  RunOptions opts;
  opts.iters = 1;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;
  opts.sep = ",";
  opts.type = Type::SEQUENTIAL;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(wide)["--wide"]("64 bit node IDs");
  cli = cli | clara::Opt(header)["--header"]("Only print CSV header, don't run");
  cli = cli | clara::Opt(overlap)["--overlap"]("Overlap I/O");
  cli = cli | clara::Opt(opts.gpus, "ids")["-g"]("gpus to use");
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

  if (overlap) {
    LOG(info, "overlapping IO and CSR");
    opts.type = Type::OVERLAPPED;
  } else {
    LOG(info, "sequential IO and CSR");
  }

  if (header) {
    print_header(opts);
    return 0;
  } else {
    if (wide) {
      LOG(debug, "using 64-bit node IDs (--wide)");
      return run<uint64_t>(opts);
    } else {
      LOG(debug, "using 32-bit node IDs (--wide for 64)");
      return run<uint32_t>(opts);
    }
  }
}
