#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_binary_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 --prefetch-async --debug -n 5"

src/benchmark-binary --header $FLAGS >> $EXPERIMENT_DIR/run.csv
for g in $GRAPH_DIR/*.bel; do
    for bs in 128; do
        echo $bs $g
        src/benchmark-binary $FLAGS --bs $bs $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done
