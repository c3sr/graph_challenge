#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_linear-mp_p

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

for g in $GRAPH_DIR/*.bel; do
    for bs in 32 64 128 256 512; do
        echo $bs $g
        echo -ne $bs,'\t' >> $EXPERIMENT_DIR/run.csv
        src/benchmark-linear-mp -g 0 --prefetch-async --debug -n 5 $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done