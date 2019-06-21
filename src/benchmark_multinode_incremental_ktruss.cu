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


  MPI_Init(&argc, &argv);

  // get the number of ranks
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (0 == rank) {
    printf("detected world size of %d\n", worldSize);
  }
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
  //LOG(info, "read_data time {}s", elapsed);
  LOG(debug, "read {} edges", edges.size());

  // create csr and count `iters` times
  std::vector<double> times;
  uint64_t nnz;
  uint64_t tris;

  // create csr
  start = std::chrono::system_clock::now();
  auto upperTriangular = [](pangolin::EdgeTy<UT> e) {
    return true; //e.first < e.second;
  };
  auto csr = pangolin::COO<UT>::from_edges(edges.begin(), edges.end(),
                                                  upperTriangular);
  LOG(debug, "nnz = {}", csr.nnz());
  elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  //LOG(info, "create CSR time {}s", elapsed);

  {
    start = std::chrono::system_clock::now();


    if (readMostly) {
        csr.read_mostly();
    }

    int gpusPerRank = gpus.size();
    int totalGPUs = worldSize * gpusPerRank;
    int *rankKBounds = (int *)malloc(3*worldSize*sizeof(int));

    ///For stream compaction //////
    int numEdges = csr.nnz();
    const size_t edgesPerGPU = (csr.nnz() + gpusPerRank - 1) / gpusPerRank;
    pangolin::Vector<UT> uSrcKp(numEdges);
    pangolin::Vector<UT> uReversed(numEdges);
    

    char *keepChar = (char*) malloc(numEdges*sizeof(char));

    // create async counters
    std::vector<pangolin::MultiGPU_Ktruss_Binary> trussCounters;
    for (int dev : gpus) {
      //LOG(info, "create device {} counter", dev);
      auto counter = pangolin::MultiGPU_Ktruss_Binary(numEdges, dev);
      counter.CreateWorkspace(numEdges);
      counter.selectedOut[0] = numEdges;
      trussCounters.push_back(counter);
      counter.InitializeWorkSpace_async(numEdges);
    }

    int edgeStart = 0;
    for (auto &counter : trussCounters) 
    { 
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t edgesToProcess = edgeStop - edgeStart;
      counter.Inialize_Unified_async(edgeStart, edgesToProcess, csr.view(), uSrcKp.data(), uReversed.data());
      edgeStart += edgesPerGPU;
    }
    uSrcKp.read_mostly();
    uReversed.read_mostly();
    
    int kmin = 3;
    int kmax = -1;

    int newKmin = kmin, newKmax=kmax;
    float factor = 0.2;
    bool startComapction = false;
    constexpr int dimBlock = 32; //For edges and nodes
    int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

    int totalReqGPUs = totalGPUs;
    int reqGPUs = gpusPerRank;
    
    while(true)
    {
      int step = 1;
      size_t edgeStart = 0;
      int gCount = 1;

      //printf("Rank = %d, New Kmin = %d, New Kmax=%d, total req gpus=%d, step=%d, local gpus=%d, cond=%d\n", rank, kmin, kmax, totalReqGPUs,step, reqGPUs, cond);
      UT numDeleted = 0;
      bool firstTry = true;
      
      for (auto &counter : trussCounters) 
      {
        counter.setDevice();
        counter.hnumaffected[0] = 1;
        CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
      }

      bool assumpAffected = true;
      dimGridEdges =  (trussCounters[0].selectedOut[0] + dimBlock - 1) / dimBlock;
      
      while(assumpAffected)
      {
        assumpAffected = false;
        for (int i=0; i<reqGPUs;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          if(counter.hnumaffected[0]>0)
          {
            /*if(counter.selectedOut[0] == numEdges)
              core_binary_direct<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gnumdeleted, 
                counter.gnumaffected, kmin + (rank*gpusPerRank+i+1)*step, 0, counter.selectedOut[0],
                csr.view(), counter.gKeep, counter.gAffected, uReversed.data(), firstTry, 2);

            else*/
              core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gDstKP, counter.gnumdeleted, 
                counter.gnumaffected, kmin + (rank*gpusPerRank+i)*step, 0, counter.selectedOut[0],
                csr.view(), counter.gKeep, counter.gAffected, uReversed.data(), firstTry, 1);
  
            //Copy to host
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumaffected, counter.gnumaffected, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumdeleted, counter.gnumdeleted, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));

            //Set gpu data to zeros
            CUDA_RUNTIME(cudaMemsetAsync(counter.gnumdeleted,0,sizeof(UT),counter.stream()));
            CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
          }
        }

        for (int i=0; i<reqGPUs;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          counter.sync();
          //printf("GPU=%d, Affected=%d, deleted=%d\n", counter.device(), counter.hnumaffected[0], counter.hnumdeleted[0]);
          assumpAffected = assumpAffected || (counter.hnumaffected[0]>0);
          counter.percentage_deleted_k = (counter.hnumdeleted[0] + numEdges - counter.selectedOut[0])*1.0/numEdges;
        }
        firstTry = false;
      }
      
      int ks = -1;
      int ke = -1;
      float prevPerc = 0;
      int fallBackGPU = -1;
      for (int i=0; i<reqGPUs;i++) 
      { 
        auto& counter = trussCounters[i];
        counter.setDevice();

        float perc = counter.perc_del_k();
        if(perc<1.0)
        {
            newKmin = ks = kmin + (rank*gpusPerRank+ i)*step;
            fallBackGPU = i;
            startComapction=true;
        }
        if(perc==1.0f)
        {
          newKmax = ke = kmin + (rank*gpusPerRank+ i)*step;
          break;
        }
      }
      //Time to all gather this
      const int dataBase = 3;
      int a[dataBase];
      a[0] = ks;
      a[1] = ke;
      a[2] = fallBackGPU;
      MPI_Allgather(a, dataBase, MPI_INT, rankKBounds,dataBase, MPI_INT, MPI_COMM_WORLD);
      MPI_Barrier( MPI_COMM_WORLD);

      bool foundKmax=false;
      int root = -1;
      for(int i= 0; i< worldSize; i++)
      {
        if(rankKBounds[dataBase*i] > -1)
        {
            newKmin=rankKBounds[dataBase*i];
            root = i;
            fallBackGPU = rankKBounds[dataBase*i + 2];

            kmax = newKmin;
        }
        
        if(!foundKmax && rankKBounds[dataBase*i + 1]>-1)
        {
          newKmax = rankKBounds[dataBase*i + 1];
          foundKmax = true;
        }
      }

      if(!foundKmax)
      {
        if(rank==root)
        {
            #pragma omp parallel for
            for (int i=0; i< numEdges; i++)
            {
                keepChar[i] = trussCounters[fallBackGPU].gKeep[i];
            }
        }

        //Now broadcast delete array to others
        if(root > -1)
        {
          int ierr = MPI_Bcast(
            //trussCounters[fallBackGPU].gKeep,
            keepChar,
            numEdges,
            MPI_CHAR,
            root,
            MPI_COMM_WORLD);
        }

        if(fallBackGPU > -1)
        {
          #pragma omp parallel for
          for (int i=0; i< numEdges; i++)
          {
              trussCounters[fallBackGPU].gKeep[i] = keepChar[i];
          }
        }
        
        for (int i=0; i<reqGPUs;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
         if(fallBackGPU != i)
          {
            counter.store_async(numEdges, trussCounters[fallBackGPU].gKeep, false);
          }
        }

        for(int i=0; i<reqGPUs; i++)
        {
          auto& counter = trussCounters[i];
          counter.setDevice();
          counter.compact(numEdges, uSrcKp.data());
          counter.sync();
          //printf("rank = %d, gpu=%d, remaining edges=%d\n", rank, i, counter.selectedOut[0]);
        }
      }
      else
        break;

      kmin += totalGPUs;
    } 


    //printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);
    for (auto &counter : trussCounters)
      counter.free();


    free(rankKBounds);

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();

    if(rank==0)
    {
      printf("%d\t%.2f\t%f\t", kmax, csr.nnz() / elapsed, elapsed);
    } 
   
  }

 
  MPI_Finalize();

  return 0;
}
