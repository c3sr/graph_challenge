#include <fmt/format.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

/*! A single-producer, single-consumer queue

    Not safe for simultaneous calls to
*/
template <typename T, size_t N = 128> struct BoundedBuffer {

  typedef T value_type;

  size_t head_;
  size_t tail_;
  size_t count_;
  std::mutex mtx_;
  std::condition_variable notFull;
  std::condition_variable notEmpty;
  T buffer_[N];

private:
  bool close_; //!< close the queue

public:
  BoundedBuffer() : close_(false), head_(0), tail_(0), count_(0) {}

  /*! \brief Pops entries off vals to fill available buffer space
   */
  void push_some(std::queue<T> &vals) {
    assert(!close_);
    std::unique_lock<std::mutex> lock(mtx_);

    // wait for the buffer to not be full
    notFull.wait(lock, [this]() { return !full(); });

    assert(!full());

    // pop enties off of vals until the buffer is full
    const size_t toPush = std::min(N - count(), vals.size());
    for (size_t i = 0; i < toPush; ++i) {
      buffer_[head_] = vals.front();
      vals.pop();
      advance_head();
    }

    lock.unlock();

    // release anyone waiting on the buffer to not be empty
    notEmpty.notify_one();

    SPDLOG_DEBUG(pangolin::logger::console, "pushed {}", toPush);
    return;
  }

  /*!
    If the buffer is not closed, blocks until there is at least one entry in the
    buffer and then returns the buffer entries.
    If the buffer is closed, zero entries are returned.
  */
  std::vector<T> pop_some() {
    std::vector<T> vals;

    std::unique_lock<std::mutex> lock(mtx_);

    // wait until the buffer is not empty or it is closed
    notEmpty.wait(lock, [this]() { return (!empty()) || close_; });

    // if the buffer is closed, this could add no elements to vals
    while (!empty()) {
      vals.emplace_back(buffer_[tail_]);
      advance_tail();
    }

    lock.unlock();

    // activate any thread waiting for the buffer to not be full
    notFull.notify_one();
    SPDLOG_DEBUG(pangolin::logger::console, "popped {}", vals.size());
    return vals;
  }

  void close() {
    close_ = true;
    // wake up everyone who might be trying to pop from the queue
    notEmpty.notify_all();
  }

  bool empty() const { return count_ == 0; }
  bool full() const { return count_ >= N; }
  size_t count() const { return count_; }
  bool closed() const { return close_; }

private:
  void advance_head() {
    head_ = (head_ + 1) % N;
    count_++;
    // SPDLOG_TRACE(pangolin::logger::console, "head -> {}, count={}", head_,
    //              count_);
  }
  void advance_tail() {
    tail_ = (tail_ + 1) % N;
    count_--;
    // SPDLOG_TRACE(pangolin::logger::console, "tail -> {}, count={}", tail_,
    //              count_);
  }
};

template <typename EDGE>
void produce(const std::string path, BoundedBuffer<EDGE> &queue) {
  pangolin::EdgeListFile file(path);

  std::vector<EDGE> fileEdges;
  std::queue<EDGE> edgeQueue;

  // while (true) {

  //   // fill up the edgeQueue
  //   size_t numRead = file.get_edges(fileEdges, 50 - edgeQueue.size());

  //   for (const auto e : fileEdges) {
  //     edgeQueue.push(e);
  //   }
  // }

  // while (!edgeQueue.empty()) {
  //   queue.push_some(edgeQueue);
  // }

  while (file.get_edges(fileEdges, 50)) {

    for (const auto e : fileEdges) {
      edgeQueue.push(e);
    }
    while (!edgeQueue.empty()) {
      queue.push_some(edgeQueue);
    }
  }

  queue.close();
}

template <typename Mat, typename EDGE>
void consume(BoundedBuffer<EDGE> &queue, Mat &mat) {

  auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first < e.second;
  };

  // keep grabbing while queue is filling
  LOG(debug, "reading queue");
  while (true) {
    std::vector<EDGE> vals = queue.pop_some();
    if (vals.empty()) {
      // the queue has no values and no more are coming, so we can quit
      assert(queue.empty());
      assert(queue.closed());
      break;
    }

    for (const auto val : vals) {
      if (upperTriangular(val)) {
        SPDLOG_TRACE(pangolin::logger::console, "{} {}", val.first, val.second);
        mat.add_next_edge(val);
      }
    }
  }

  mat.finish_edges();
}

int main(int argc, char **argv) {

  typedef uint64_t Index64;
  typedef pangolin::EdgeTy<Index64> Edge64;

  std::vector<int> gpus;
  std::string path;
  int iters = 1;
  bool help = false;
  bool debug = false;
  bool verbose = false;

  bool readMostly = false;
  bool accessedBy = false;
  bool prefetchAsync = false;

  clara::Parser cli;
  cli = cli | clara::Help(help);
  cli = cli | clara::Opt(debug)["--debug"]("print debug messages to stderr");
  cli = cli |
        clara::Opt(verbose)["--verbose"]("print verbose messages to stderr");
  cli = cli | clara::Opt(gpus, "ids")["-g"]("gpus to use");
  cli = cli | clara::Opt(readMostly)["--read-mostly"](
                  "mark data as read-mostly by all gpus before kernel");
  cli = cli | clara::Opt(accessedBy)["--accessed-by"](
                  "mark data as accessed-by all GPUs before kernel");
  cli = cli | clara::Opt(prefetchAsync)["--prefetch-async"](
                  "prefetch data to all GPUs before kernel");
  cli = cli | clara::Opt(iters, "N")["-n"]("number of counts");
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
    prefetchAsync = false;
    LOG(warn, "disabling prefetch");
    readMostly = false;
    LOG(warn, "disabling readMostly");
    accessedBy = false;
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

  // read data / build
  auto start = std::chrono::system_clock::now();

  BoundedBuffer<Edge64> queue;
  pangolin::COO<Index64> csr;
  // start a thread to read the matrix data
  LOG(debug, "start disk reader");
  std::thread reader(produce<Edge64>, path, std::ref(queue));

  // start a thread to build the matrix
  LOG(debug, "start csr build");
  std::thread builder(consume<pangolin::COO<Index64>, Edge64>, std::ref(queue),
                      std::ref(csr));
  // consume(queue, csr, &readerActive);

  LOG(debug, "waiting for disk reader...");
  reader.join();
  LOG(debug, "waiting for CSR builder...");
  builder.join();
  assert(queue.empty());

  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data/build time {}s", elapsed);

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;

  for (int i = 0; i < iters; ++i) {
    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (readMostly) {
      csr.read_mostly();
      for (const auto &gpu : gpus) {
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "read-mostly CSR time {}s", elapsed);

    // accessed-by
    nvtxRangePush("accessed-by");
    start = std::chrono::system_clock::now();
    if (accessedBy) {
      for (const auto &gpu : gpus) {
        csr.accessed_by(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "accessed-by CSR time {}s", elapsed);

    // prefetch
    nvtxRangePush("prefetch");
    start = std::chrono::system_clock::now();
    if (prefetchAsync) {
      for (const auto &gpu : gpus) {
        csr.prefetch_async(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "prefetch CSR time {}s", elapsed);

    // count triangles
    nvtxRangePush("count");
    start = std::chrono::system_clock::now();

    // create async counters
    std::vector<pangolin::LinearTC> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(pangolin::LinearTC(dev));
    }

    // determine the number of edges per gpu
    const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    size_t edgeStart = 0;
    for (auto &counter : counters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(),
          numEdges);
      counter.count_async(csr.view(), numEdges, edgeStart);
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
    nvtxRangePop();
    LOG(info, "count time {}s", elapsed);
    LOG(info, "{} triangles ({} teps)", total, csr.nnz() / elapsed);
    times.push_back(elapsed);
    tris = total;
    nnz = csr.nnz();
  }

  std::cout << path << ",\t" << nnz << ",\t" << tris;
  for (const auto &t : times) {
    std::cout << ",\t" << t;
  }
  std::cout << std::endl;

  return 0;
}
