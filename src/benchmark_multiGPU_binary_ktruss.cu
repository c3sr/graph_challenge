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
  LOG(info, "read_data time {}s", elapsed);
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
  LOG(info, "create CSR time {}s", elapsed);

  // read-mostly
  nvtxRangePush("read-mostly");
  start = std::chrono::system_clock::now();
  if (readMostly) {
    for (const auto &gpu : gpus) {
      csr.read_mostly();
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
  //start = std::chrono::system_clock::now();

  //Find Kmax
  //This is part of file reading
  std::map<UT, int> degree;
  for (int i = 0; i < csr.num_rows(); i++)
  {
    int start = csr.rowPtr_[i];
    int end = csr.rowPtr_[i + 1];

    if (degree.count(end - start) == 0)
      degree[end - start] = 0;

    degree[end - start]++;
  }

  {
    start = std::chrono::system_clock::now();
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
      LOG(info, "create device {} counter", dev);
      auto counter = pangolin::MultiGPU_Ktruss_Binary(numEdges, dev);
      counter.CreateWorkspace(numEdges);
      trussCounters.push_back(counter);
      counter.InitializeWorkSpace_async(csr.view(), numEdges);
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
    int kmax = getMaxK(degree);


    if(rank==0)
    {
      printf("# GPUs=%d\n", totalGPUs);
      printf("Kmin=%d, kmax=%d\n", kmin, kmax);
    } 
    int newKmin = kmin, newKmax=kmax;
    float factor = 0.5;
    while(kmax-kmin > 1)
    {

      int totalReqGPUs = (kmax - kmin) > totalGPUs? totalGPUs:(kmax-kmin-1);
      int reqGPUs = 0;
      int cond = totalReqGPUs - (rank)*gpusPerRank;
      if(cond > gpusPerRank)
        reqGPUs = gpusPerRank;
      else if(cond>0)
          reqGPUs = cond;

      if(reqGPUs == 0)
      {
        printf("Rank=%d is not needed any more\n", rank);
        kmax = kmin;
        newKmax = kmin;
      }

      int step = (kmax*factor-kmin)/(totalReqGPUs + 1);
      factor = 1.0;
      size_t edgeStart = 0;
      int gCount = 1;

      printf("Rank = %d, New Kmin = %d, New Kmax=%d, total req gpus=%d, step=%d, local gpus=%d, cond=%d\n", rank, kmin, kmax, totalReqGPUs,step, reqGPUs, cond);
      
      constexpr int dimBlock = 32; //For edges and nodes
      int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
      UT numDeleted = 0;
      bool firstTry = true;
      
      for (auto &counter : trussCounters) 
      {
        counter.setDevice();
        counter.hnumaffected[0] = 1;
        CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
        counter.selectedOut[0] = numEdges;
      }

      bool assumpAffected = true;
      dimGridEdges =  (trussCounters[0].selectedOut[0] + dimBlock - 1) / dimBlock;
      
      for(int i=0; i<reqGPUs; i++)
      {
        int pk = kmin + (rank*gpusPerRank+i+1)*step;
        printf("Rank=%d, GPU=%d, k=%d\n", rank, i, pk);
      }

      while(assumpAffected)
      {
        assumpAffected = false;
        for (int i=0; i<reqGPUs;i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          if(counter.hnumaffected[0]>0)
          {
            //printf("GPU=%d, k=%d\n", counter.device(), kmin+(i+1)*step);
            core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gDstKP, counter.gnumdeleted, 
              counter.gnumaffected, kmin + (rank*gpusPerRank+i+1)*step, 0, counter.selectedOut[0],
              csr.view(), counter.gKeep, counter.gAffected, uReversed.data(), firstTry, 2);

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
          counter.percentage_deleted_k = (counter.hnumdeleted[0])*1.0/numEdges;
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
            newKmin = ks = kmin + (rank*gpusPerRank+ i + 1)*step;
            fallBackGPU = i;
        }
        if(perc==1.0f && prevPerc != 1.0)
        {
          newKmax = ke = kmin + (rank*gpusPerRank+ i + 1)*step;
        }
        prevPerc = perc;
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
        }
        
        if(!foundKmax && rankKBounds[dataBase*i + 1]>-1)
        {
          newKmax = rankKBounds[dataBase*i + 1];
          foundKmax = true;
        }

        //printf("From rank=%d, rank=%d [%d,%d] fallback=%d\t", rank, i, rankKBounds[dataBase*i], rankKBounds[dataBase*i+1], fallBackGPU);
      }
      //printf("\n");

     if(rank==root)
      {
        int sum=0;
        for (int i=0; i< numEdges; i++)
        {
            //sum += trussCounters[fallBackGPU].gKeep[i];
            keepChar[i] = trussCounters[fallBackGPU].gKeep[i];
        }
        //printf("ROOT=%d, remaining edges = %d\n", rank, sum);
      }

      //Now broadcast delete array to others
      if(root > -1)
      {
        //printf("rank = %d, Let us broadcast: fallBackGPU=%d, root=%d\n", rank, fallBackGPU, root);
        //broadcast :D
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
        UT sum=0;
        for (int i=0; i< numEdges; i++)
        {
            trussCounters[fallBackGPU].gKeep[i] = keepChar[i];
            //sum += trussCounters[fallBackGPU].gKeep[i];
        }
        //printf("rank=%d, remaining edges = %u\n", rank, sum);
      } 

      //MPI_Barrier( MPI_COMM_WORLD);

      for (int i=0; i<reqGPUs;i++) 
      { 
        auto& counter = trussCounters[i];
        counter.setDevice();
        if(fallBackGPU == -1)
        {
          counter.rewind_async(numEdges);
        }
        else if(fallBackGPU != i)
        {
          counter.store_async(numEdges, trussCounters[fallBackGPU].gKeep);
        }
        else
        {
          counter.store_async(numEdges);
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

      kmin = newKmin;
      kmax = newKmax;
    } 

    //printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);
    for (auto &counter : trussCounters)
      counter.free();


    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "count time {}s", elapsed);
    LOG(info, "Rank={} MOHA {} ktruss ({} teps)", rank, kmin, csr.nnz() / elapsed);
    times.push_back(elapsed);
    //tris = total;
    nnz = csr.nnz();

    //std::cout << path << ",\t" << nnz << ",\t" << tris;
    for (const auto &t : times) {
      std::cout << ",\t" << t;
    }
    std::cout << std::endl;
    free(rankKBounds);
  }

 
  MPI_Finalize();

  return 0;
}
