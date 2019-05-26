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
  uint64_t tris;
  for (int i = 0; i < iters; ++i) {
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

    omp_set_num_threads(152);
    // count triangles
    nvtxRangePush("count");
    //start = std::chrono::system_clock::now();

    //Find Kmax
    std::map<UT, int> degree;
    for (int i = 0; i < csr.num_rows(); i++)
    {
      int start = csr.rowPtr_[i];
      int end = csr.rowPtr_[i + 1];
  
      if (degree.count(end - start) == 0)
        degree[end - start] = 0;
  
      degree[end - start]++;
    }

    

    start = std::chrono::system_clock::now();

    int numEdges = csr.nnz();
    // create async counters
    std::vector<pangolin::MultiGPU_Ktruss_Binary> trussCounters;
    for (int dev : gpus) {
      LOG(info, "create device {} counter", dev);
      auto counter = pangolin::MultiGPU_Ktruss_Binary(numEdges, dev);
      trussCounters.push_back(counter);
      counter.InitializeWorkSpace_async(csr.view(), numEdges);
    }
   
    int kmin = 3;
    int kmax = getMaxK(degree);
    //Attempt different k to bound kmin and kmax
    printf("# GPUs=%d\n", gpus.size());

    int newKmin = kmin, newKmax=kmax;
    int factor = 2;
    while(kmax-kmin > gpus.size())
    {
      int step = (kmax/factor-kmin)/(gpus.size() + 1);
      factor=1;
      size_t edgeStart = 0;
      int gCount = 1;

     
      constexpr int dimBlock = 32; //For edges and nodes
      int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
      UT numDeleted = 0;
      bool firstTry = true;
      
      for (auto &counter : trussCounters) 
       {
         counter.setDevice();
         counter.hnumaffected[0] = 1;
         CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
       }

      bool assumpAffected = true;
      dimGridEdges =  (numEdges + dimBlock - 1) / dimBlock;
      
      while(assumpAffected)
      {
        assumpAffected = false;
        for (int i=0; i<gpus.size();i++) 
        { 
          auto& counter = trussCounters[i];
          counter.setDevice();
          //if(counter.hnumaffected[0]>0)
          {

            //printf("GPU=%d, k=%d\n", i+2, kmin+(i+1)*step);
            core_binary_direct<dimBlock><<<dimGridEdges,dimBlock,0,counter.stream()>>>(counter.gnumdeleted, 
              counter.gnumaffected, kmin + (i+1)*step, 0, numEdges,
              csr.view(), counter.gKeep, counter.gAffected, counter.gReveresed, firstTry, 4); //<Tunable: 4>

            //Copy to host
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumaffected, counter.gnumaffected, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));
            CUDA_RUNTIME(cudaMemcpyAsync(counter.hnumdeleted, counter.gnumdeleted, sizeof(UT), cudaMemcpyDeviceToHost, counter.stream()));

            //Set gpu data to zeros
           // CUDA_RUNTIME(cudaMemsetAsync(counter.gnumdeleted,0,sizeof(UT),counter.stream()));
            //CUDA_RUNTIME(cudaMemsetAsync(counter.gnumaffected,0,sizeof(UT),counter.stream()));
            
            counter.zero();
          

          }
        }

        for (auto &counter : trussCounters) 
        {
          counter.setDevice();
          counter.sync();


          //printf("GPU=%d, Affected=%d, deleted=%d\n", counter.device(), counter.hnumaffected[0], counter.hnumdeleted[0]);
          assumpAffected = assumpAffected || (counter.hnumaffected[0]>0);
          counter.percentage_deleted_k = (counter.hnumdeleted[0])*1.0/numEdges;

        }

          firstTry = false;
          
      
      }
     
      gCount = 1;
      float prevPerc = 0;
      for (auto &counter : trussCounters) 
      {
        counter.setDevice();
        counter.sync();

        float perc = counter.perc_del_k();
        //deletePercentage.push_back(perc);
        //printf("Percentage = %f\n", perc);

        if(perc<1.0)
        {
            newKmin = kmin + gCount*step;

            //Store for this gpu: as a begining
            counter.store_async(numEdges);
        }
        else
        {
          counter.rewind_async(numEdges);
        }
        
        if(perc==1.0f && prevPerc != 1.0)
        {
          newKmax = kmin + gCount*step;
        }

        prevPerc = perc;
        gCount++;
      }

      kmin = newKmin;
      kmax = newKmax;
     //printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);
  }

  printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);

/*



    printf("New Kmin = %d, New Kmax=%d\n", newKmin, newKmax);

    kmin=newKmin;
    kmax=newKmax;

    //Real Deal: Ktruss
   
    const size_t edgesPerGPU = (csr.nnz() + gpus.size() - 1) / gpus.size();
    LOG(info, "{} edges per GPU", edgesPerGPU);
    
    edgeStart = 0;
    for (auto &counter : trussCounters) 
    { 
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t edgesToProcess = edgeStop - edgeStart;
      //counter.InitializeArrays_async(edgeStart, edgesToProcess, csr.view(), mKeep.data(), mAffected.data(), mReversed.data(), mPrevKeep.data(), mSrcKP.data(), mDestKP.data());
      counter.InitializeWorkSpace_async(csr.view(), numEdges);
      edgeStart += edgesPerGPU;

      printf("Inialized Part on input\n");
    }
    for (auto &counter : trussCounters) 
    {
      counter.sync();
    }


    printf("Let us start ktruss\n");

    UT k;
		int originalKmin = kmin;
		int originalKmax = kmax;
    bool cond = kmax - kmin > 1;
    float minPercentage = 0.5;
    bool firstTry=true;
    UT numDeleted=0;
    bool stillAffected = true;
    float percDeleted = 0;
    int uMax=2;
    while (cond)
		{
			k =  kmin*minPercentage + kmax*(1-minPercentage);
      firstTry = true;
      numDeleted = 0;
      stillAffected = true;
		
      while(stillAffected)
      {
        stillAffected = false;
        numDeleted = 0;
        size_t edgeStart = 0;
        int gc = 0;
        for (auto &counter : trussCounters) {
          const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
          const size_t edgesToProcess = edgeStop - edgeStart;
          LOG(info, "start async count on GPU {} ({} edges) k = {}", counter.device(),
            edgesToProcess, k);
          counter.core_gpu_async(
            mDestKP.data(),
            k,  
            edgeStart,
            edgesToProcess, 
            csr.view(), 
            mReversed.data(), 
            firstTry, 
            uMax);

          edgeStart += edgesPerGPU;
          gc++;
        }
        for (auto &counter : trussCounters) 
        {
          counter.sync();
          
          UT ndel = counter.numDeleted();
          UT naff = counter.numAffected();
          numDeleted += ndel;
          if(naff>0)
            stillAffected = true;

          printf("%d, %d\n", ndel, naff);
        }
        //CPU data movements
        omp_set_num_threads(152);
        #pragma omp parallel for
        for(int i=0; i<numEdges; i++)
        {
          bool finalKeep = true;
          bool finalAffected = false;
          for (auto &counter : trussCounters)
          {
            finalKeep = finalKeep && counter.gKeep[i];
            finalAffected = finalAffected || counter.gAffected[i];
          }

          for (auto &counter : trussCounters)
          {
            counter.gKeep[i] = finalKeep;
            counter.gAffected[i] = finalAffected;
          }
        }
  
          //////////


        //Copy Effects


        firstTry = false;
      }


      percDeleted= (numDeleted)*1.0/numEdges;
      printf("Exited inner loop, Precentage Deleted at k=%d equals %f\n", k, percDeleted);
      break; 







			cond = kmax - kmin > 1;
		}

		k= k= numDeleted==numDeleted? k-1:k;



    /*size_t edgeStart = 0;
    for (auto &counter : trussCounters) {
      const size_t edgeStop = std::min(edgeStart + edgesPerGPU, csr.nnz());
      const size_t numEdges = edgeStop - edgeStart;
      LOG(debug, "start async count on GPU {} ({} edges)", counter.device(),
          numEdges);
      counter.findKtrussBinary_async(3, maxK, csr.view(), csr.num_rows(), numEdges,0,edgeStart);
      edgeStart += edgesPerGPU;
    }*/


   
    // wait for counting operations to finish
    /*uint64_t total = 0;
    for (auto &counter : counters) {
      LOG(debug, "wait for counter on GPU {}", counter.device());
      counter.sync();
      total += counter.count();
    }*/

    elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(info, "count time {}s", elapsed);
    //LOG(info, "MOHA {} ktruss ({} teps)", total, csr.nnz() / elapsed);
    times.push_back(elapsed);
    //tris = total;
    nnz = csr.nnz();
  }

  //std::cout << path << ",\t" << nnz << ",\t" << tris;
  for (const auto &t : times) {
    std::cout << ",\t" << t;
  }
  std::cout << std::endl;

  return 0;
}
