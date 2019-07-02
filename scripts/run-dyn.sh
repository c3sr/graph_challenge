#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_dyn

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 --prefetch-async --debug -n 5"

src/benchmark-dyn $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
for g in $GRAPH_DIR/*.bel; do
    for sb in 0 0.25 0.5 1 2 1e10; do
        echo $g $sb
        src/benchmark-dyn $FLAGS --scale-binary $sb $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done