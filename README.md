# graph_challenge

## Getting Started

```
git clone git@github.com:c3sr/graph_challenge.git --recursive
cd graph_challenge
mkdir build
cd build
cmake ..
make
```

If you are building on a Power9 system using your own install of clang5, do something like

```
# so CMake can configure OpenMP correctly
export LD_LIBRARY_PATH=~/software/llvm-5.0.0/lib
# toolchain file tells nvcc to use clang
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=`readlink -f ../thirdparty/pangolin/cmake/clang.toolchain`
```

If you are planning to develop in Pangolin, you should check out a new branch of pangolin to use

```
cd thirdparty/pangolin
git checkout -b my-branch
```

## Datasets

Some dataset information is here:

https://graphchallenge-datasets.netlify.com

Graphs may be downloaded using [cwpearson/graph-datasets2](https://github.com/cwpearson/graph-datasets) (preferred) or [cwpearson/graph-datasets](https://github.com/cwpearson/graph-datasets).

```
mkdir ~/graphs
python graph-datasets/tools/download.py --out ~/graphs
```

There are many graphs. You can restrict the graphs that are downloaded.

To speed up I/O, it is recommended to convert the graphs to a binary format

```
python graph-datasets/tools/convert.py ~/graphs/graphtsv -t bel
```

## Citing

Our 2018 submission may be cited with the following bibtex entry

    @inproceedings{mailthody2018collaborative,
    title={Collaborative (CPU+ GPU) Algorithms for Triangle Counting and Truss Decomposition},
    author={Mailthody, Vikram S and Date, Ketan and Qureshi, Zaid and Pearson, Carl and Nagi, Rakesh and Xiong, Jinjun and Hwu, Wen-mei},
    booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
    pages={1--7},
    year={2018},
    organization={IEEE}
    }

Our 2019 submissions may be cited with the following bibtex entry

    @inproceedings{mailthody2018collaborative,
    title={Collaborative (CPU+ GPU) Algorithms for Triangle Counting and Truss Decomposition},
    author={Mailthody, Vikram S and Date, Ketan and Qureshi, Zaid and Pearson, Carl and Nagi, Rakesh and Xiong, Jinjun and Hwu, Wen-mei},
    booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
    pages={1--7},
    year={2018},
    organization={IEEE}
    }