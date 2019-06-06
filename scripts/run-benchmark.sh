#! env bash

set -eou pipefail

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_binary

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

for b in $GRAPH_DIR/*.bel; do
echo $b
src/benchmark-binary -g 0 --prefetch-async --debug -n 4 $b >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log 
done