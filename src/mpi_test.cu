#if PANGOLIN_USE_MPI == 1
#include <mpi.h>
#endif

#include "pangolin/algorithm/fill.cuh"
#include "pangolin/dense/buffer.cuh"
#include "pangolin/init.hpp"

int main(int argc, char **argv) {
  pangolin::init();

#if PANGOLIN_USE_MPI == 1
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

  // create an int
  pangolin::Buffer<int> v(1);

  // use the GPU to set it to 1
  pangolin::device_fill(v.data(), v.size(), 1);
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // use MPI to do a reduction
  int reduction = -1;
  MPI_Reduce(v.data(), &reduction, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // print on the rank 0 node
  if (0 == rank) {
    printf("%d\n", reduction);
  }

  // Finalize the MPI environment.
  MPI_Finalize();
  return 0;
#else
  (void)argc;
  (void)argv;
  printf("pangolin not compiled with MPI support, or MPI not found\n");
  return -1;
#endif
}
