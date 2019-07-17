#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_dyn_g01_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 -g 1 --prefetch-async --read-mostly --debug -n 5"

src/benchmark-dyn $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
for g in $GRAPH_DIR/*.bel; do
    for sb in 2; do
        echo $g $sb
        numactl -p 0 src/benchmark-dyn $FLAGS --scale-binary $sb $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done
