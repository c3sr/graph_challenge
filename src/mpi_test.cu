#include <mpi.h>

#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

int main(int argc, char **argv) {


  int kk = 10;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // get the number of ranks
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (0 == rank) {
    printf("detected world size of %d\n", worldSize);
  }

  printf("KK=%d", kk);

  // create an int
  //pangolin::Buffer<int> v(1);

  // use the GPU to set it to 1
  //pangolin::device_fill(v.data(), v.size(), 1);
  //CUDA_RUNTIME(cudaDeviceSynchronize());

  printf("Rank=%d\n", rank);
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Rank=%d, Device Number: %d\n", rank, i);
    /*printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);*/

    printf("  Bus ID: %d\n", prop.pciBusID);
    printf("  PCI ID: %d\n", prop.pciDeviceID);
  }



  int v[1];
  v[0] = 10;

  // use MPI to do a reduction
  int reduction = -1;
  MPI_Reduce(v, &reduction, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // print on the rank 0 node
  if (0 == rank) {
    printf("%d\n", reduction);
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}