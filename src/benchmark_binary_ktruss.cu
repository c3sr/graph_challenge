#include <fmt/format.h>
#include <iostream>

#include<map>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

#include "pangolin/algorithm/ktruss_binary_HSC.cuh"

#define UT uint32_t


int getMaxK(std::map<UT, int> degree)
{
	typedef std::map<UT, int>::reverse_iterator  it_type;
	int maxK = 0;
	int reverseCount = 0;
	bool getNext = false;
	for (it_type m = degree.rbegin(); m != degree.rend(); m++)
	{
		int degree = m->first;
		int proposedKmax = degree + 2;

		reverseCount += m->second;

		if (reverseCount >= proposedKmax)
		{
			maxK = proposedKmax;
			break;
		}
	}

	return maxK;

}


int main(int argc, char **argv) {


  pangolin::init(); 

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

  // read data
  auto start = std::chrono::system_clock::now();
  pangolin::EdgeListFile file(path);

  std::vector<pangolin::DiEdge<UT>> edges;
  std::vector<pangolin::DiEdge<UT>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;
  for (int i = 0; i < iters; ++i) {
    // create csr
    start = std::chrono::system_clock::now();
    auto upperTriangular = [](pangolin::DiEdge<UT> e) {
      return true; //e.first < e.second;
    };
    auto csr = pangolin::CSRCOO<UT>::from_edges(edges.begin(), edges.end(),
                                                   upperTriangular);
    LOG(debug, "nnz = {}", csr.nnz());
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(info, "create CSR time {}s", elapsed);

    // read-mostly
    nvtxRangePush("read-mostly");
    start = std::chrono::system_clock::now();
    if (readMostly) {
      for (const auto &gpu : gpus) {
        //csr.read_mostly(gpu);
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
        //csr.accessed_by(gpu);
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
        //csr.prefetch_async(gpu);
        CUDA_RUNTIME(cudaSetDevice(gpu));
        CUDA_RUNTIME(cudaDeviceSynchronize());
      }
    }
    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "prefetch CSR time {}s", elapsed);

    // count triangles
    nvtxRangePush("count");
    //start = std::chrono::system_clock::now();

    //Find Kmax
    int count = 0;

    for(int i=0; i< csr.nnz(); i++)
    {
      UT src = csr.rowInd_[i];
      UT dst = csr.colInd_[i];

      if(src<0 || src > csr.num_rows())
        {
          printf("at element %d of rowIndex, val = %d where num of rows = %d\n", i, src, csr.num_rows());
          count++;
          if(count >= 6)
            break;
        }
      
        if(dst <0 ||  dst > csr.num_rows())
        {
          printf("at element %d of colIndex, val = %d where num of rows = %d\n", i, dst, csr.num_rows());
          count++;
          if(count >= 6)
            break;
        }  
    }



    std::map<UT, int> degree;
    for (int i = 0; i < csr.num_rows(); i++)
    {
      int start = csr.rowPtr_[i];
      int end = csr.rowPtr_[i + 1];
  
      if (degree.count(end - start) == 0)
        degree[end - start] = 0;
  
      degree[end - start]++;
    }

    // create async counters
    std::vector<pangolin::SingleGPU_Ktruss_Binary> counters;
    for (int dev : gpus) {
      LOG(debug, "create device {} counter", dev);
      counters.push_back(pangolin::SingleGPU_Ktruss_Binary(csr.nnz() ,dev));
    }

    // determine the number of edges per gpu
    const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
    LOG(debug, "{} edges per GPU", edgesPerGPU);

    // launch counting operations
    start = std::chrono::system_clock::now();

    UT *rowPtr = csr.rowPtr_.data();
    UT *rowInd = csr.rowInd_.data();
    UT *colInd = csr.colInd_.data();

    int maxK = getMaxK(degree);

    printf("NNZ=%u, NR=%u,Upper bound kmax=%d\n", csr.nnz(), csr.num_rows(), maxK);


    size_t edgeStart = 0;
    for (auto &counter : counters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(),
          numEdges);
      counter.findKtrussBinary_hsc_async(3, maxK, rowPtr, rowInd,colInd, csr.num_rows(), numEdges,0,edgeStart);
      //counter.findKtrussBinary_async(3, maxK, csr.view(), csr.num_rows(), numEdges,0,edgeStart);
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
    LOG(info, "MOHA {} ktruss ({} teps)", total, csr.nnz() / elapsed);
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
