# graph_challenge

## Datasets

Graphs may be downloaded from https://graphchallenge.mit.edu/data-sets.
Alternatively, `python tools/get_data.py` may be used to download some datasets.

```
mkdir ~/graphs
python tools/get_data.py --out ~/graphs
```

Datasets may be converted to a binary edge list format using `python tools/tsv-to-bel.py`.

To produce graph.bel, try one of the following:
```
python tools/tsv-to-bel.py graph.tsv
python tools/tsv-to-bel.py graph.tsv graph.bel
```


## Citing

Our 2017 submission may be cited with the following bibtex entry

    @inproceedings{mailthody2018collaborative,
    title={Collaborative (CPU+ GPU) Algorithms for Triangle Counting and Truss Decomposition},
    author={Mailthody, Vikram S and Date, Ketan and Qureshi, Zaid and Pearson, Carl and Nagi, Rakesh and Xiong, Jinjun and Hwu, Wen-mei},
    booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
    pages={1--7},
    year={2018},
    organization={IEEE}
    }