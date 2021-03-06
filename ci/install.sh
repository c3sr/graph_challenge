set -x -e

source ci/env.sh

# Install Cmake if it doesn't exist
mkdir -p $CMAKE_PREFIX
if [[ ! -f $CMAKE_PREFIX/bin/cmake ]]; then
    if [[ $TRAVIS_CPU_ARCH == "ppc64le" ]]; then
        wget -qSL https://cmake.org/files/v3.13/cmake-3.13.5.tar.gz -O cmake.tar.gz
        tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
        rm cmake.tar.gz
        cd $CMAKE_PREFIX
        ./bootstrap --prefix=$CMAKE_PREFIX
        make -j `nproc` install
    elif [[ $TRAVIS_CPU_ARCH == "amd64" ]]; then
        wget -qSL https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.tar.gz -O cmake.tar.gz
        tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
        rm cmake.tar.gz
    fi
fi
cd $HOME

sudo apt-get update 

if [[ $USE_NUMA == "1" ]]; then
sudo apt-get install -y --no-install-recommends \
  libnuma-dev
fi

if [[ $USE_MPI == "1" ]]; then
sudo apt-get install -y --no-install-recommends \
  libopenmpi-dev openmpi-bin
fi

## install CUDA
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

if [[ $TRAVIS_CPU_ARCH == "ppc64le" && $CUDA_VERSION == "102" && $TRAVIS_DIST == "bionic" ]]; then
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/ppc64el/cuda-repo-ubuntu1804_10.2.89-1_ppc64el.deb"
elif [[ $TRAVIS_CPU_ARCH == "amd64" && $CUDA_VERSION == "102" && $TRAVIS_DIST == "bionic" ]]; then
    CUDA_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb"
fi

wget -SL $CUDA_URL -O cuda.deb

sudo dpkg -i cuda.deb
sudo apt-get update 
sudo apt-get install -y --no-install-recommends \
  cuda-toolkit-10-2