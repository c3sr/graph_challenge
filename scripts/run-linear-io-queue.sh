#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_linear-io-queue_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 --prefetch-async --debug -n 4"

src/benchmark-linear-io-queue $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log

for g in $GRAPH_DIR/*.bel; do
    for bs in 512; do
        echo $bs $g
        src/benchmark-linear-io-queue --bs $bs $FLAGS $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done
