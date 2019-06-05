/*!

Count triangles using the per-edge linear search.
Overlap IO and matrix construction with a queue of vectors of edges

*/

#include <fmt/format.h>
#include <iostream>
#include <mutex>
#include <thread>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

/*! A single-producer, single-consumer queue

    Not safe for simultaneous calls to
*/
template <typename T, size_t N = 128> struct LockQueue {

  typedef T value_type;

  volatile size_t head_;
  volatile size_t tail_;
  volatile size_t count_;
  std::mutex mtx_;
  T buffer_[N];

  LockQueue() : head_(0), tail_(0), count_(0) {}

  void push(const T &val) {
    std::lock_guard<std::mutex> lock(mtx_);
    assert(count_ < N);
    buffer_[head_] = val;
    advance_head();
    return;
  }

  T pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    assert(count_ > 0);
    T val = buffer_[tail_];
    advance_tail();
    return val;
  }

  bool empty() { return count_ == 0; }
  bool full() { return count_ >= N; }
  size_t count() { return count_; }

private:
  void advance_head() {
    head_ = (head_ + 1) % N;
    count_++;
    SPDLOG_TRACE(pangolin::logger::console(), "head -> {}, count={}", head_,
                 count_);
  }
  void advance_tail() {
    tail_ = (tail_ + 1) % N;
    count_--;
    SPDLOG_TRACE(pangolin::logger::console(), "tail -> {}, count={}", tail_,
                 count_);
  }
};

/*! A multi-producer, multi-consumer queue

    full if head_ + 1 == tail_
    empty if head_ == tail_

    \tparam N the number of slots in the buffer. N+1 slots are allocated, one
   is wasted
*/
template <typename T, size_t N = 127> struct LockQueue2 {

  typedef T value_type;

  static constexpr size_t BUF_SIZE = N + 1;
  volatile size_t head_;
  volatile size_t tail_;
  std::mutex mtx_;
  T buffer_[BUF_SIZE];

  LockQueue2() : head_(1), tail_(0) {}

  // returns false if queue is full
  bool push(const T &val) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (unsafe_full()) {
      return false;
    }

    buffer_[head_] = val;
    advance_head();
    return true;
  }

  // returns number of inserted elements
  size_t push_many(const std::vector<T> &vals) {
    size_t i = 0;
    {
      std::lock_guard<std::mutex> lock(mtx_);

      // insert as many elements as possible

      for (i = 0; !unsafe_full(); ++i, advance_head()) {
        buffer_[head_] = vals[i];
      }
    }

    // erase inserted elements
    vals.erase(vals.begin(), vals.begin() + i);

    return i;
  }

  // returns false if queue was empty
  bool pop(T &t) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (unsafe_empty()) {
      return false;
    }
    t = buffer_[tail_];
    advance_tail();
    return true;
  }

  // returns number of popped elements
  size_t pop_many(std::vector<T> &vals) {
    vals.clear();
    vals.reserve(unsafe_count());
    {
      std::lock_guard<std::mutex> lock(mtx_);

      // pop as many elements as possible
      for (size_t i = 0; !unsafe_empty(); ++i, advance_tail()) {
        vals.push_back(buffer_[tail_]);
      }
    }

    return vals.size();
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(mtx_);
    return unsafe_empty();
  }
  bool full() {
    std::lock_guard<std::mutex> lock(mtx_);
    return unsafe_full();
  }
  size_t count() {
    std::lock_guard<std::mutex> lock(mtx_);
    return unsafe_count();
  }

private:
  bool unsafe_count() { return (head_ - tail_) % BUF_SIZE; }
  bool unsafe_full() { return (head_ + 1) % BUF_SIZE == tail_; }
  bool unsafe_empty() { return head_ == tail_; }

  void advance_head() {
    head_ = (head_ + 1) % BUF_SIZE;
    SPDLOG_TRACE(pangolin::logger::console(), "head -> {}, count={}", head_,
                 unsafe_count());
  }
  void advance_tail() {
    tail_ = (tail_ + 1) % BUF_SIZE;
    SPDLOG_TRACE(pangolin::logger::console(), "tail -> {}, count={}", tail_,
                 unsafe_count());
  }
};

template <typename EDGE>
void produce(const std::string path, LockQueue<std::vector<EDGE>> &queue,
             volatile bool *busy) {
  *busy = true;
  pangolin::EdgeListFile file(path);

  std::vector<EDGE> fileEdges;
  while (file.get_edges(fileEdges, 50)) {

    // wait for queue to not be full
    while (queue.full()) {
      // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    queue.push(fileEdges);
  }
  *busy = false;
}

template <typename Mat, typename EDGE>
void consume(
    LockQueue<std::vector<EDGE>> &queue, Mat &mat,
    const volatile bool *filling //!< [in] is the queue still being filled?
) {

  auto upperTriangular = [](pangolin::EdgeTy<uint64_t> e) {
    return e.first < e.second;
  };

  // keep grabbing while queue is filling
  LOG(debug, "reading queue while filling");
  while (*filling) {
    if (!queue.empty()) {
      std::vector<EDGE> vals = queue.pop();
      for (auto &val : vals) {
        if (upperTriangular(val)) {
          SPDLOG_TRACE(pangolin::logger::console(), "{} {}", val.first,
                       val.second);
          mat.add_next_edge(val);
        }
      }
    }
  }
  // drain queue
  LOG(debug, "draining queue after filling stops");
  while (!queue.empty()) {
    std::vector<EDGE> vals = queue.pop();
    for (auto &val : vals) {
      if (upperTriangular(val)) {
        SPDLOG_TRACE(pangolin::logger::console(), "drain {} {}", val.first,
                     val.second);
        mat.add_next_edge(val);
      }
    }
  }
  mat.finish_edges();
}

int main(int argc, char **argv) {

  pangolin::init();

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

  LockQueue<std::vector<Edge64>> queue;
  pangolin::COO<Index64> csr;
  volatile bool readerActive = true;
  // start a thread to read the matrix data
  LOG(debug, "start disk reader");
  std::thread reader(produce<Edge64>, path, std::ref(queue), &readerActive);

  // start a thread to build the matrix
  LOG(debug, "start csr build");
  std::thread builder(consume<pangolin::COO<Index64>, Edge64>, std::ref(queue),
                      std::ref(csr), &readerActive);
  // consume(queue, csr, &readerActive);

  LOG(debug, "waiting for disk reader...");
  reader.join();
  readerActive = false;
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
      counter.count_async(csr.view(), edgeStart, numEdges);
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
