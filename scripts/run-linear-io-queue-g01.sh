#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_linear-io-queue-g01_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="--bs 512 -g 0 -g 1 --prefetch-async --debug -n 5"

src/benchmark-linear-io-queue $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log

for g in $GRAPH_DIR/*.bel; do
    echo $g
    numactl -p 0 src/benchmark-linear-io-queue $FLAGS $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
done