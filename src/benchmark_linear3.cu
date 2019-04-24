#include <fmt/format.h>
#include <iostream>
#include <map>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

using pangolin::BoundedBuffer;

template <typename EDGE>
void produce(const std::string path,
             std::map<int, BoundedBuffer<EDGE>> &NUMAToQueue) {
  pangolin::EdgeListFile file(path);

  std::vector<EDGE> fileEdges;
  std::deque<EDGE> edgeQueue;

  while (file.get_edges(fileEdges, 50)) {

    // add the edges to each queue
    for (auto &kv : NUMAToQueue) {
      auto &queue = kv.second;
      for (const auto e : fileEdges) {
        edgeQueue.push_back(e);
      }
      while (!edgeQueue.empty()) {
        queue.push_some(edgeQueue);
      }
    }
  }

  for (auto &kv : NUMAToQueue) {
    auto &queue = kv.second;
    queue.close();
  }
}

template <typename Mat, typename EDGE>
void consume(const int numa, BoundedBuffer<EDGE> &queue, Mat &mat) {

  // bind to NUMA node if available
  if (pangolin::numa::available()) {
    pangolin::numa::set_strict();
    pangolin::numa::bind(numa);
  }

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

  // unbind when done
  if (pangolin::numa::available()) {
    pangolin::numa::unbind();
  }
}

/*!
  Create one unified-memory data structure in each NUMA region close to the used
  GPUs.
*/
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

  // associate a numa region with each GPU
  // arbitrarily choose the first numa region that has affinity with each GPU
  // each NUMA region may have multiple GPUs, but each GPU has only a single
  // NUMA region
  std::map<int, int> GPUToNUMA;
  std::map<int, std::set<int>> NUMAToGPUs;
  for (const auto &gpu : gpus) {
    std::set<int> cpus = pangolin::topology::device_cpu_affinity(gpu);
    std::set<int> numas = pangolin::topology::cpu_numa_affinity(cpus);
    assert(!numas.empty() && "expect to find at least one region");
    int numa = *numas.begin();
    LOG(debug, "GPU {} -> NUMA {}", gpu, numa);
    GPUToNUMA.insert(std::make_pair(gpu, numa));

    if (NUMAToGPUs.count(numa) == 0) {
      NUMAToGPUs[numa] = std::set<int>();
    }
    NUMAToGPUs[numa].insert(gpu);
    LOG(debug, "NUMA {} -> GPU {}", numa, gpu);
  }

  // create a queue for each numa region that has a GPU. A single producer will
  // read from disk and insert edges into multiple consumer queues
  std::map<int, pangolin::BoundedBuffer<Edge64>> NUMAToQueue;
  std::map<int, pangolin::COO<Index64>> NUMAToCSR;
  for (auto &kv : NUMAToGPUs) {
    int numa = kv.first;
    NUMAToQueue.insert(std::make_pair(numa, BoundedBuffer<Edge64>()));
    NUMAToCSR.insert(std::make_pair(numa, pangolin::COO<Index64>()));
  }

  // read data / build
  auto start = std::chrono::system_clock::now();

  // start a thread to read the matrix data
  LOG(debug, "start disk reader");
  std::thread reader(produce<Edge64>, path, std::ref(NUMAToQueue));

  // start one thread per numa region to build the COOs
  LOG(debug, "start csr builders");
  std::vector<std::thread> builders;
  for (auto &kv : NUMAToQueue) {
    int numa = kv.first;
    LOG(debug, "start mat builder for numa {}", numa);
    auto &queue = kv.second;
    auto &csr = NUMAToCSR[numa];
    builders.push_back(std::thread(consume<pangolin::COO<Index64>, Edge64>,
                                   numa, std::ref(queue), std::ref(csr)));
  }

  LOG(debug, "waiting for disk reader...");
  reader.join();
  LOG(debug, "waiting for CSR builders...");
  for (auto &builder : builders) {
    builder.join();
  }

  for (auto &kv : NUMAToQueue) {
    auto &queue = kv.second;
    assert(queue.closed());
    assert(queue.empty());
  }

  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data/build time {}s", elapsed);

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;

  for (int i = 0; i < iters; ++i) {
    /*
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
    */
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
    // all mats are the same, so just use the first
    const auto &mat = NUMAToCSR.begin()->second;
    const size_t edgesPerGPU = (mat.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    size_t edgeStart = 0;
    for (auto &counter : counters) {

      // determine the right CSR for this GPU by looking up which NUMA region
      // the GPU is using
      const int numa = GPUToNUMA[counter.device()];
      auto &csr = NUMAToCSR[numa];
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
    LOG(info, "{} triangles ({} teps)", total, mat.nnz() / elapsed);
    times.push_back(elapsed);
    tris = total;
    nnz = mat.nnz();
  }

  std::cout << path << ",\t" << nnz << ",\t" << tris;
  for (const auto &t : times) {
    std::cout << ",\t" << t;
  }
  std::cout << std::endl;

  return 0;
}
