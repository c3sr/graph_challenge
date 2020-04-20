#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_io-overlap_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 --prefetch-async --debug -n 5"

src/benchmark-io-overlap $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log

for g in $GRAPH_DIR/*.bel; do
  echo $g
  src/benchmark-io-overlap $FLAGS $g           >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
  src/benchmark-io-overlap $FLAGS $g --overlap >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
done
