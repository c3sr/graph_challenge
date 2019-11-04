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



  edges.clear();
  edges.shrink_to_fit();
  fileEdges.clear();
  fileEdges.shrink_to_fit();

  
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
    UT *rowPtr = csr.rowPtr_.data();
    UT *rowInd = csr.rowInd_.data();
    UT *colInd = csr.colInd_.data();


    UT numEdges = csr.nnz();
    int numGpus = gpus.size();
    int numNodes = csr.num_nodes();

    UT edgesPerGPU = (numEdges + numGpus - 1) / numGpus;
    UT *uSrcKp, *uDstKp, *uReversed;

    printf("GPus=%d, NNZ=%u, nr=%u\n", numGpus, numEdges, csr.num_rows());

    CUDA_RUNTIME(cudaMallocManaged((void **) &uSrcKp, numEdges*sizeof(UT)));
    CUDA_RUNTIME(cudaMallocManaged((void **) &uDstKp, numEdges*sizeof(UT)));
    CUDA_RUNTIME(cudaMallocManaged((void **) &uReversed, numEdges*sizeof(UT)));


    CUDA_RUNTIME(cudaMemAdvise(rowInd, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
    CUDA_RUNTIME(cudaMemAdvise(colInd, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));
    CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));


    // create async counters
    std::vector<pangolin::MultiGPU_Ktruss_Incremental> trussCounters;
    for (int dev : gpus) {
      LOG(info, "create device {} counter", dev);
      auto counter = pangolin::MultiGPU_Ktruss_Incremental(numEdges, dev);
      counter.CreateWorkspace(numEdges);
      trussCounters.push_back(counter);
      counter.InitializeWorkSpace_async(numEdges);
    }

    UT edgeStart = 0;
    for (auto &counter : trussCounters) 
    { 
      counter.selectedOut[0] = numEdges;

      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
      const size_t edgesToProcess = edgeStop - edgeStart;
      counter.Inialize_Unified_async(edgeStart, edgesToProcess, rowPtr, rowInd, colInd, uSrcKp, uReversed);
      edgeStart += edgesPerGPU;
    }

    UT *ptrSrc, *ptrDst;
    UT *s1, *d1, *s2, *d2;

    s1 = rowInd;
		d1 = colInd;

		s2 = uSrcKp;
		d2 = uDstKp;

		ptrSrc = s1;
		ptrDst = d1;
    
    int kmin = 3;
    int kmax=-1;
    constexpr int dimBlock = 32; //For edges and nodes
    int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

 
    CUDA_RUNTIME(cudaMemAdvise(uReversed, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 /* ignored */));

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
            core_full_direct<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gnumdeleted, 
                counter.gnumaffected, kmin+i, 0, numEdges,
                rowPtr, ptrSrc, ptrDst, counter.gKeep, counter.gAffected, uReversed, firstTry, 1);
  
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

          LOG(info, "Device={}, k={}, Deleted={}", i, kmin, counter.hnumdeleted[0]);
        }
        firstTry = false;
        CUDA_RUNTIME(cudaGetLastError());
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
      if(!foundKmax)
      {
        printf("Start Stream Compaction\n");
        auto& c = trussCounters[fallBackGPU];
        float percDeleted = (c.hnumdeleted[0])*1.0/numEdges;
        if(percDeleted > 0.1)
        {
          
          UT     *d_temp_storage0=NULL, *d_temp_storage1=NULL, *d_temp_storage2=NULL, *d_temp_storage3 = NULL;
          size_t   temp_storage_bytes0 = 0, temp_storage_bytes1 = 0, temp_storage_bytes2 = 0, temp_storage_bytes3 = 0;

          UT edgeStart = 0;
          UT edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          UT edgesToProcess = edgeStop - edgeStart;
          trussCounters[0].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage0, temp_storage_bytes0,  &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]), trussCounters[0].selectedOut, edgesPerGPU, trussCounters[0].stream());
          
          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[1].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage1, temp_storage_bytes1,  &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]), trussCounters[1].selectedOut, edgesPerGPU, trussCounters[1].stream());

          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[2].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage2, temp_storage_bytes2,  &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]), trussCounters[2].selectedOut, edgesPerGPU, trussCounters[2].stream());

          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[3].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage3, temp_storage_bytes3,  &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]), trussCounters[3].selectedOut, edgesPerGPU, trussCounters[3].stream());

          CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage0, temp_storage_bytes0));
          CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage1, temp_storage_bytes1));
          CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage2, temp_storage_bytes2));
          CUDA_RUNTIME(cudaMallocManaged(&d_temp_storage3, temp_storage_bytes3));
          CUDA_RUNTIME(cudaGetLastError());

          printf("Created Temp storage\n");

          edgeStart = 0;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[0].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage0, temp_storage_bytes0, &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]),  trussCounters[0].selectedOut, edgesToProcess,  trussCounters[0].stream());
          cub::DevicePartition::Flagged(d_temp_storage0, temp_storage_bytes0, &(d1[edgeStart]), &(c.gKeep[edgeStart]), &(d2[edgeStart]),  trussCounters[0].selectedOut, edgesToProcess,  trussCounters[0].stream());
          trussCounters[0].sync();

          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[1].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage1, temp_storage_bytes1, &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]),  trussCounters[1].selectedOut, edgesToProcess,  trussCounters[1].stream());
          cub::DevicePartition::Flagged(d_temp_storage1, temp_storage_bytes1, &(d1[edgeStart]), &(c.gKeep[edgeStart]), &(d2[edgeStart]),  trussCounters[1].selectedOut, edgesToProcess,  trussCounters[1].stream());
          trussCounters[1].sync();

          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[2].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage2, temp_storage_bytes2, &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]),  trussCounters[2].selectedOut, edgesToProcess,  trussCounters[2].stream());
          cub::DevicePartition::Flagged(d_temp_storage2, temp_storage_bytes2, &(d1[edgeStart]), &(c.gKeep[edgeStart]), &(d2[edgeStart]),  trussCounters[2].selectedOut, edgesToProcess,  trussCounters[2].stream());
          trussCounters[2].sync();


          edgeStart += edgesPerGPU;
          edgeStop = std::min(edgeStart + edgesPerGPU, numEdges);
          edgesToProcess = edgeStop - edgeStart;
          trussCounters[3].setDevice();
          cub::DevicePartition::Flagged(d_temp_storage3, temp_storage_bytes3, &(s1[edgeStart]), &(c.gKeep[edgeStart]), &(s2[edgeStart]),  trussCounters[3].selectedOut, edgesToProcess,  trussCounters[3].stream());
          cub::DevicePartition::Flagged(d_temp_storage3, temp_storage_bytes3, &(d1[edgeStart]), &(c.gKeep[edgeStart]), &(d2[edgeStart]),  trussCounters[3].selectedOut, edgesToProcess,  trussCounters[3].stream());
          trussCounters[3].sync();
          
          CUDA_RUNTIME(cudaFree(d_temp_storage0));
          CUDA_RUNTIME(cudaFree(d_temp_storage1));
          CUDA_RUNTIME(cudaFree(d_temp_storage2));
          CUDA_RUNTIME(cudaFree(d_temp_storage3));
          CUDA_RUNTIME(cudaGetLastError());

          /*for (int i=0; i<numGpus;i++) 
          { 
            auto& counter = trussCounters[i];
            counter.setDevice();
            counter.sync();
          }*/

          printf("Finished CUB PARTITION\n");
          CUDA_RUNTIME(cudaGetLastError());

          CUDA_RUNTIME(cudaMemAdvise(s1, numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          CUDA_RUNTIME(cudaMemAdvise(d1, numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          edgeStart = 0;
          UT dstStart = 0;
          for (int i=0; i<numGpus;i++) 
          { 
            auto& counter = trussCounters[i];
            counter.setDevice();
            counter.MoveData_async(edgeStart, dstStart, counter.selectedOut[0], s2, s1 , d2, d1);
            edgeStart += edgesPerGPU;
            dstStart += counter.selectedOut[0];
            printf("Remaining edges by GPU %d is %d\n", i, counter.selectedOut[0]);
          }
          numEdges=0;
          for (int i=0; i<numGpus;i++) 
          { 
            auto& counter = trussCounters[i];
            counter.setDevice();
            counter.sync();
            numEdges += counter.selectedOut[0];
          }
          printf("----Remaining Edges=%d\n", numEdges);
          CUDA_RUNTIME(cudaMemAdvise(s1, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0));
          CUDA_RUNTIME(cudaMemAdvise(d1, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0));
          
         
          CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          CUDA_RUNTIME(cudaMemAdvise(uReversed, numEdges * sizeof(UT), cudaMemAdviseUnsetReadMostly, 0));
          
          edgesPerGPU = (numEdges + numGpus - 1) / numGpus;
          dimGridEdges =  (numEdges + dimBlock - 1) / dimBlock;
          
          c.setDevice();
          RebuildArrays<dimBlock><<<dimGridEdges,dimBlock,0,c.stream()>>>(0, numEdges, numEdges, rowPtr, ptrSrc); 
          RebuildReverse<dimBlock><<<dimGridEdges,dimBlock,0,c.stream()>>>(0, numEdges, rowPtr, ptrSrc, ptrDst, uReversed);
          
          printf("Finished Row Pointer and Reverse\n");
          CUDA_RUNTIME(cudaGetLastError());
          
          
          for (auto &counter : trussCounters)
          { 
            counter.setDevice();
            counter.InitializeWorkSpace_async(numEdges);
          } 

          printf("Finished Reinialize Keep and Affected\n");
          CUDA_RUNTIME(cudaGetLastError());

          CUDA_RUNTIME(cudaMemAdvise(rowPtr, (numNodes+1) * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));
          CUDA_RUNTIME(cudaMemAdvise(uReversed, numEdges * sizeof(UT), cudaMemAdviseSetReadMostly, 0 ));

          for (int i=0; i<numGpus;i++) 
          { 
            auto& counter = trussCounters[i];
            counter.setDevice();
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

    cudaFree(uSrcKp);
    cudaFree(uDstKp);
    cudaFree(uReversed);
  

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
