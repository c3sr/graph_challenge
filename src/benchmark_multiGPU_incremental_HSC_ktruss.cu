#include <mpi.h>

#include <fmt/format.h>
#include <iostream>

#include "omp.h"

#include<map>

#include <nvToolsExt.h>

#include "clara/clara.hpp"
#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"
#include "pangolin/algorithm/zero.cuh"

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
		int proposedKmax = degree + 1;

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
  pangolin::Config config;

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

  std::vector<pangolin::EdgeTy<UT>> edges;
  std::vector<pangolin::EdgeTy<UT>> fileEdges;
  while (file.get_edges(fileEdges, 10)) {
    edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
  }
  double elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;

  // create csr
  start = std::chrono::system_clock::now();
  auto upperTriangular = [](pangolin::EdgeTy<UT> e) {
    return true; //e.first < e.second;
  };
  auto csr = pangolin::COO<UT>::from_edges(edges.begin(), edges.end(),
                                                  upperTriangular);
  LOG(debug, "nnz = {}", csr.nnz());
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  LOG(info, "create CSR time {}s", elapsed);

  
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
  //start = std::chrono::system_clock::now();

  {
    start = std::chrono::system_clock::now();

    //csr.read_mostly();
    UT *rowPtr = csr.rowPtr_.data();
    UT *rowInd = csr.rowInd_.data();
    UT *colInd = csr.colInd_.data();

    int numEdges = csr.nnz();
    int numGpus = gpus.size();
    int numNodes = csr.num_nodes();

    int edgesPerGPU = (numEdges + numGpus - 1) / numGpus;
    pangolin::Vector<UT> uSrcKp(numEdges);
    pangolin::Vector<UT> uDstKp(numEdges);
    pangolin::Vector<UT> uReversed(numEdges);
    
    printf("NNZ=%d\n", numEdges);

    // create async counters
    std::vector<pangolin::MultiGPU_Ktruss_Incremental> trussCounters;
    for (int dev : gpus) {
      LOG(info, "create device {} counter", dev);
      auto counter = pangolin::MultiGPU_Ktruss_Incremental(numEdges, dev);
      counter.CreateWorkspace(numEdges);
      trussCounters.push_back(counter);
      counter.InitializeWorkSpace_async(numEdges);
    }

    int edgeStart = 0;
    for (auto &counter : trussCounters) 
    { 

      counter.selectedOut[0] = numEdges;

      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
      const size_t edgesToProcess = edgeStop - edgeStart;
      counter.Inialize_Unified_async(edgeStart, edgesToProcess, rowPtr, rowInd, colInd, uSrcKp.data(), uReversed.data());
      edgeStart += edgesPerGPU;
    }

    UT *ptrSrc, *ptrDst;
    UT *s1, *d1, *s2, *d2;

    s1 = rowInd;
		d1 = colInd;

		s2 = uSrcKp.data();
		d2 = uDstKp.data();

		ptrSrc = s1;
		ptrDst = d1;
    
    int kmin = 3;
    int kmax=-1;
    constexpr int dimBlock = 32; //For edges and nodes
    int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

    if(numGpus > 1)
    {
      CUDA_RUNTIME(cudaMemAdvise(ptrSrc, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
      CUDA_RUNTIME(cudaMemAdvise(ptrDst, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
      CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
      CUDA_RUNTIME(cudaMemAdvise(uReversed.data(), numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
    }

    while(true)
    {

      LOG(info, "kmin={}, remaining edges={}", kmin, numEdges);

      bool firstTry = true;
      for (auto &counter : trussCounters) 
      {
        counter.setDevice();
        counter.hnumaffected[0] = 1;
        CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
      }

      bool assumpAffected = true;
      
      /*nvtxRangePush("kernel per k");
      start = std::chrono::system_clock::now();*/
      while(assumpAffected)
      {
        assumpAffected = false;
        for (int i=0; i<numGpus;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          if(counter.hnumaffected[0]>0)
          {
              core_direct<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gnumdeleted, 
                counter.gnumaffected, kmin+i, 0, numEdges,
                rowPtr, ptrSrc, ptrDst, counter.gKeep, counter.gAffected, uReversed.data(), firstTry, 1);
  
            //Copy to host
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumaffected, counter.gnumaffected, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumdeleted, counter.gnumdeleted, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));

            //Set gpu data to zeros
            CUDA_RUNTIME(cudaMemsetAsync(counter.gnumdeleted,0,sizeof(UT),counter.stream()));
            CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
          }
        }
       
        for (int i=0; i<numGpus;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          counter.sync();
          assumpAffected = assumpAffected || (counter.hnumaffected[0]>0);
          counter.percentage_deleted_k = (counter.hnumdeleted[0])*1.0/numEdges;
        }
        firstTry = false;
      }
  

      bool foundKmax = false;
      int fallBackGPU = -1;
      for (int i=0; i<numGpus;i++) 
      { 
        auto& counter = trussCounters[i];
        counter.setDevice();

        if(numEdges - counter.hnumdeleted[0] > 0)
        {
          kmax = kmin + i;
          fallBackGPU = i;

        }
        else
        {
          foundKmax = true;
          break;
        }
      }

      

      kmin += numGpus;
      int counter = 0; 
     
      if(!foundKmax)
      {
        auto& c = trussCounters[fallBackGPU];
        float percDeleted = (c.hnumdeleted[0])*1.0/numEdges;
        if(c.hnumdeleted[0] > 1000)
        {
          if(numGpus > 1)
          {
            CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
            CUDA_RUNTIME(cudaMemAdvise(uReversed.data(), numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          }


          //each gpu stores latest keep
          
          c.setDevice();
          void     *d_temp_storage = NULL;
          size_t   temp_storage_bytes = 0;
          
          cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, s1, c.gKeep, s2, c.selectedOut, numEdges, c.stream());
          CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
          cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, s1, c.gKeep, s2, c.selectedOut, numEdges, c.stream());
          cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d1, c.gKeep, d2, c.selectedOut, numEdges, c.stream());
          CUDA_RUNTIME(cudaFree(d_temp_storage));

          cudaDeviceSynchronize();
          CUDA_RUNTIME(cudaGetLastError());

          if(numGpus > 1)
          {
            CUDA_RUNTIME(cudaMemAdvise(s1, numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
            CUDA_RUNTIME(cudaMemAdvise(d1, numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          
            CUDA_RUNTIME(cudaMemAdvise(s2, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));
            CUDA_RUNTIME(cudaMemAdvise(d2, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));
          }

          numEdges = c.selectedOut[0];
          edgesPerGPU = (numEdges + numGpus - 1) / numGpus;
          dimGridEdges =  (numEdges + dimBlock - 1) / dimBlock;

          ptrSrc = s2;
          s2 = s1;
          s1 = ptrSrc;

          ptrDst = d2;
          d2 = d1;
          d1 = ptrDst;
          
          c.setDevice();
          RebuildArrays<dimBlock><<<dimGridEdges,dimBlock,0,c.stream()>>>(0, numEdges, numEdges, rowPtr, ptrSrc); 
          RebuildReverse<dimBlock><<<dimGridEdges,dimBlock,0,c.stream()>>>(0, numEdges, rowPtr, ptrSrc, ptrDst, uReversed.data());
          for (auto &counter : trussCounters)
          { 
            counter.setDevice();
            counter.InitializeWorkSpace_async(numEdges);
          } 

          
          if(numGpus > 1)
          {
            CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));
            CUDA_RUNTIME(cudaMemAdvise(uReversed.data(), numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));
          }

          CUDA_RUNTIME(cudaGetLastError());
          

          for (auto &counter : trussCounters)
          { 
            counter.sync();
            
          } 
        }
      }
      else{
        break;
      }
    } 

    //printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);
    for (auto &counter : trussCounters)
      counter.free();

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "count time {}s", elapsed);
    LOG(info, "MOHA {} ktruss ({} teps)", kmax, csr.nnz() / elapsed);
    times.push_back(elapsed);
    //tris = total;
    nnz = csr.nnz();

    //std::cout << path << ",\t" << nnz << ",\t" << tris;
    for (const auto &t : times) {
      std::cout << ",\t" << t;
    }
    std::cout << std::endl;
  }

  return 0;
}
