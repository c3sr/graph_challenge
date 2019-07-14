/*!

Count triangles using the linear search.
Simultaneously read and build the CSR with a queue of edges.

*/

#include <iostream>
#include <thread>
#include <vector>

#include <clara/clara.hpp>
#include <fmt/format.h>

#include <nvToolsExt.h>

#include "pangolin/algorithm/tc_edge_linear.cuh"
#include "pangolin/bounded_buffer.hpp"
#include "pangolin/configure.hpp"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_coo.hpp"

// Buffer is a BoundedBuffer with two entries (double buffer)
template <typename T> using Buffer = pangolin::BoundedBuffer<T, 2>;

template <typename V> void print_vec(const V &vec, const std::string &sep) {
  for (const auto &e : vec) {
    fmt::print("{}{}", sep, e);
  }
}

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

template <typename Mat> void consume(Buffer<std::vector<typename Mat::edge_type>> &queue, Mat &mat) {
  typedef typename Mat::index_type Index;
  typedef typename Mat::edge_type Edge;

  double queueTime = 0, csrTime = 0;
  auto upperTriangular = [](const Edge &e) { return e.first < e.second; };

  // keep grabbing while queue is filling
  Index maxNode = 0;
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
  int iters;
  std::vector<int> gpus;
  std::string path;
  std::string sep;
  int blockSize;

  bool readMostly;
  bool accessedBy;
  bool prefetchAsync;
  bool preCountBarrier;
};

void print_header(const RunOptions &opts) {
  fmt::print("benchmark{0}bs{0}gpus{0}graph{0}nodes{0}edges{0}tris", opts.sep);
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

template <typename Index> int run(RunOptions &opts) {
  typedef pangolin::EdgeTy<Index> Edge;
  using pangolin::RcStream;

  auto gpus = opts.gpus;
  if (gpus.empty()) {
    LOG(warn, "no GPUs provided on command line, using GPU 0");
    gpus.push_back(0);
  }

  // create a stream for each GPU
  std::vector<RcStream> streams;
  for (const auto &gpu : gpus) {
    streams.push_back(RcStream(gpu));
  }

  // Check for unified memory support
  bool managed = true;
  for (auto gpu : gpus) {
    cudaDeviceProp prop;
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, gpu));
    // We check for concurrentManagedAccess, as devices with only the
    // managedAccess property have extra synchronization requirements.
    if (prop.concurrentManagedAccess) {
      LOG(debug, "device {} prop.concurrentManagedAccess=1", gpu);
    } else {
      LOG(warn, "device {} prop.concurrentManagedAccess=0", gpu);
    }
    managed = managed && prop.concurrentManagedAccess;
    if (prop.canMapHostMemory) {
      LOG(debug, "device {} prop.canMapHostMemory=1", gpu);
    } else {
      LOG(warn, "device {} prop.canMapHostMemory=0", gpu);
    }
  }

  if (!managed) {
    opts.prefetchAsync = false;
    LOG(warn, "disabling prefetch");
    opts.readMostly = false;
    LOG(warn, "disabling readMostly");
    opts.accessedBy = false;
    LOG(warn, "disabling accessedBy");
  }

  // Check for mapping host memory memory support
  for (auto gpu : gpus) {
    CUDA_RUNTIME(cudaSetDevice(gpu));
    unsigned int flags = 0;
    CUDA_RUNTIME(cudaGetDeviceFlags(&flags));
    if (flags & cudaDeviceMapHost) {
      LOG(debug, "device {} cudaDeviceMapHost=1", gpu);
    } else {
      LOG(warn, "device {} cudaDeviceMapHost=0", gpu);
      // exit(-1);
    }
  }

  // count `iters` times
  std::vector<double> totalTimes;
  std::vector<double> gpuTimes;
  std::vector<double> countTimes;
  std::vector<double> kernelTimes;
  uint64_t nnz;
  uint64_t numRows;
  uint64_t tris;
  for (int i = 0; i < opts.iters; ++i) {

    // read data / build
    const auto totalStart = std::chrono::system_clock::now();

    Buffer<std::vector<Edge>> queue;
    pangolin::CSRCOO<Index> csr;
    // start a thread to read the matrix data
    LOG(debug, "start disk reader");
    std::thread reader(produce<Edge>, opts.path, std::ref(queue));
    // start a thread to build the matrix
    LOG(debug, "start csr build");
    std::thread builder(consume<pangolin::CSRCOO<Index>>, std::ref(queue), std::ref(csr));
    // consume(queue, csr, &readerActive);
    LOG(debug, "waiting for disk reader...");
    reader.join();
    LOG(debug, "waiting for CSR builder...");
    builder.join();
    assert(queue.empty());
    double elapsed = (std::chrono::system_clock::now() - totalStart).count() / 1e9;
    LOG(info, "read_data/build time {}s", elapsed);
    LOG(debug, "CSR: nnz = {}, rows = {}", csr.nnz(), csr.num_rows());

    const auto gpuStart = std::chrono::system_clock::now();

    // read-mostly
    nvtxRangePush("read-mostly");
    if (opts.readMostly) {
      csr.read_mostly();
    }
    elapsed = (std::chrono::system_clock::now() - gpuStart).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);

    // accessed-by
    nvtxRangePush("accessed-by");
    auto start = std::chrono::system_clock::now();
    if (opts.accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "accessed-by CSR time {}s", elapsed);

    // prefetch
    nvtxRangePush("prefetch");
    start = std::chrono::system_clock::now();
    if (opts.prefetchAsync) {
      for (size_t gpuIdx = 0; gpuIdx < gpus.size(); ++gpuIdx) {
        auto &gpu = gpus[gpuIdx];
        cudaStream_t stream = streams[gpuIdx].stream();
        csr.prefetch_async(gpu, stream);
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "prefetch CSR time {}s", elapsed);

    if (opts.preCountBarrier) {
      LOG(debug, "sync streams after hints");
      for (auto stream : streams) {
        stream.sync();
      }
    }

    // count triangles
    nvtxRangePush("count");
    const auto countStart = std::chrono::system_clock::now();

    // create async counters
    std::vector<pangolin::LinearTC> counters;
    for (size_t gpuIdx = 0; gpuIdx < gpus.size(); ++gpuIdx) {
      auto dev = gpus[gpuIdx];
      auto &stream = streams[gpuIdx];
      LOG(debug, "create device {} counter", dev);
      counters.push_back(pangolin::LinearTC(dev, stream));
    }

    // determine the number of edges per gpu
    const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    size_t edgeStart = 0;
    for (auto &counter : counters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(), numEdges);
      counter.count_async(csr.view(), edgeStart, numEdges, opts.blockSize);
      edgeStart += edgesPerGPU;
    }

    // wait for counting operations to finish
    uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }
    const auto stop = std::chrono::system_clock::now();
    nvtxRangePop(); // count
    LOG(info, "{} triangles", total);

    // record graph stats
    tris = total;
    nnz = csr.nnz();
    numRows = csr.num_rows();

    // record times
    const double totalElapsed = (stop - totalStart).count() / 1e9;
    const double gpuElapsed = (stop - gpuStart).count() / 1e9;
    const double countElapsed = (stop - countStart).count() / 1e9;
    LOG(info, "total time {}s ({} teps)", totalElapsed, nnz / totalElapsed);
    LOG(info, "gpu time   {}s ({} teps)", gpuElapsed, nnz / gpuElapsed);
    LOG(info, "count time {}s ({} teps)", countElapsed, nnz / countElapsed);
    totalTimes.push_back(totalElapsed);
    gpuTimes.push_back(gpuElapsed);
    countTimes.push_back(countElapsed);

    for (auto &counter : counters) {
      double secs = counter.kernel_time();
      int dev = counter.device();
      LOG(info, "gpu {} kernel time {}s ({} teps)", dev, secs, nnz / secs);
    }
    if (counters.size() == 1) {
      kernelTimes.push_back(counters[0].kernel_time());
    } else {
      kernelTimes.push_back(0);
    }
  }

  if (opts.iters > 0) {
    fmt::print("linear-io-queue");
    fmt::print("{}{}", opts.sep, opts.blockSize);
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

  // options not passed to run
  bool help = false;
  bool debug = false;
  bool verbose = false;
  bool wide = false;
  bool header = false;

  // options passed to run
  RunOptions opts;
  opts.iters = 1;
  opts.readMostly = false;
  opts.accessedBy = false;
  opts.prefetchAsync = false;
  opts.blockSize = 512;
  opts.sep = ",";
  opts.preCountBarrier = true;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli | clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(wide)["--wide"]("64 bit node IDs");
  cli = cli | clara::Opt(header)["--header"]("Only print CSV header, don't run");
  cli = cli | clara::Opt(opts.gpus, "ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(opts.readMostly)["--read-mostly"]("mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(opts.accessedBy)["--accessed-by"]("mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(opts.prefetchAsync)["--prefetch-async"]("prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(opts.iters, "N")["-n"]("number of counts");
  cli = cli | clara::Opt(opts.blockSize, "N")["--bs"]("block size for counting kernel");
  cli = cli | clara::Opt(opts.preCountBarrier)["--barrier"]("barrier before count kernels");
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
