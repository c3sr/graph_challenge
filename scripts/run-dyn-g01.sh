#! env bash

set -eou pipefail

DONE="
/data/graph_challenge/201512012345.v18571154_e38040320.bel
/data/graph_challenge/201512020000.v35991342_e74485420.bel
/data/graph_challenge/201512020030.v68863315_e143414960.bel
/data/graph_challenge/201512020130.v128568730_e270234840.bel
/data/graph_challenge/201512020330.v226196185_e480047894.bel
/data/graph_challenge/A2a.bel
/data/graph_challenge/amazon0302_adj.bel
/data/graph_challenge/amazon0312_adj.bel
/data/graph_challenge/amazon0505_adj.bel
/data/graph_challenge/amazon0601_adj.bel
/data/graph_challenge/as20000102_adj.bel
/data/graph_challenge/ca-CondMat_adj.bel
/data/graph_challenge/ca-GrQc_adj.bel
/data/graph_challenge/ca-HepPh_adj.bel
/data/graph_challenge/ca-HepTh_adj.bel
/data/graph_challenge/cit-HepPh_adj.bel
/data/graph_challenge/cit-HepTh_adj.bel
/data/graph_challenge/cit-Patents_adj.bel
/data/graph_challenge/email-Enron_adj.bel
/data/graph_challenge/email-EuAll_adj.bel
/data/graph_challenge/facebook_combined_adj.bel
/data/graph_challenge/flickrEdges_adj.bel
/data/graph_challenge/friendster_adj.bel
/data/graph_challenge/graph500-scale18-ef16_adj.bel
/data/graph_challenge/graph500-scale19-ef16_adj.bel
/data/graph_challenge/graph500-scale20-ef16_adj.bel
/data/graph_challenge/graph500-scale21-ef16_adj.bel
/data/graph_challenge/graph500-scale22-ef16_adj.bel
/data/graph_challenge/graph500-scale23-ef16_adj.bel
/data/graph_challenge/graph500-scale24-ef16_adj.bel
/data/graph_challenge/graph500-scale25-ef16_adj.bel
/data/graph_challenge/loc-brightkite_edges_adj.bel
/data/graph_challenge/loc-gowalla_edges_adj.bel
/data/graph_challenge/oregon1_010331_adj.bel
/data/graph_challenge/oregon1_010407_adj.bel
/data/graph_challenge/oregon1_010414_adj.bel
/data/graph_challenge/oregon1_010421_adj.bel
/data/graph_challenge/oregon1_010428_adj.bel
/data/graph_challenge/oregon1_010505_adj.bel
/data/graph_challenge/oregon1_010512_adj.bel
/data/graph_challenge/oregon1_010519_adj.bel
/data/graph_challenge/oregon1_010526_adj.bel
/data/graph_challenge/oregon2_010331_adj.bel
/data/graph_challenge/oregon2_010407_adj.bel
/data/graph_challenge/oregon2_010414_adj.bel
/data/graph_challenge/oregon2_010421_adj.bel
/data/graph_challenge/oregon2_010428_adj.bel
/data/graph_challenge/oregon2_010505_adj.bel
/data/graph_challenge/oregon2_010512_adj.bel
/data/graph_challenge/oregon2_010519_adj.bel
/data/graph_challenge/oregon2_010526_adj.bel
/data/graph_challenge/P1a.bel
/data/graph_challenge/p2p-Gnutella04_adj.bel
/data/graph_challenge/p2p-Gnutella05_adj.bel
/data/graph_challenge/p2p-Gnutella06_adj.bel
/data/graph_challenge/p2p-Gnutella08_adj.bel
/data/graph_challenge/p2p-Gnutella09_adj.bel
/data/graph_challenge/p2p-Gnutella24_adj.bel
/data/graph_challenge/p2p-Gnutella25_adj.bel
/data/graph_challenge/p2p-Gnutella30_adj.bel
/data/graph_challenge/p2p-Gnutella31_adj.bel
/data/graph_challenge/roadNet-CA_adj.bel
/data/graph_challenge/roadNet-PA_adj.bel
/data/graph_challenge/roadNet-TX_adj.bel
/data/graph_challenge/soc-Epinions1_adj.bel
/data/graph_challenge/soc-Slashdot0811_adj.bel
/data/graph_challenge/soc-Slashdot0902_adj.bel
/data/graph_challenge/Theory-16-25-81-B1k.bel
/data/graph_challenge/Theory-16-25-81-B2k.bel
/data/graph_challenge/Theory-16-25-81-Bk.bel
/data/graph_challenge/Theory-16-25-B1k.bel
/data/graph_challenge/Theory-16-25-B2k.bel
/data/graph_challenge/Theory-16-25-Bk.bel
/data/graph_challenge/Theory-256-625-B1k.bel
/data/graph_challenge/Theory-256-625-B2k.bel
/data/graph_challenge/Theory-256-625-Bk.bel
/data/graph_challenge/Theory-25-81-256-B1k.bel
/data/graph_challenge/Theory-25-81-256-B2k.bel
/data/graph_challenge/Theory-25-81-256-Bk.bel
/data/graph_challenge/Theory-25-81-B1k.bel
/data/graph_challenge/Theory-25-81-B2k.bel
/data/graph_challenge/Theory-25-81-Bk.bel
"
REMAIN="
Theory-3-4-5-9-16-25-B1k.bel
Theory-3-4-5-9-16-25-B2k.bel
Theory-3-4-5-9-16-25-Bk.bel
Theory-3-4-5-9-16-B1k.bel
Theory-3-4-5-9-16-B2k.bel
Theory-3-4-5-9-16-Bk.bel
Theory-3-4-5-9-B1k.bel
Theory-3-4-5-9-B2k.bel
Theory-3-4-5-9-Bk.bel
Theory-3-4-5-B1k.bel
Theory-3-4-5-B2k.bel
Theory-3-4-5-Bk.bel
Theory-3-4-B1k.bel
Theory-3-4-B2k.bel
Theory-3-4-Bk.bel
Theory-4-5-9-16-25-B1k.bel
Theory-4-5-9-16-25-B2k.bel
Theory-4-5-9-16-25-Bk.bel
Theory-4-5-9-16-B1k.bel
Theory-4-5-9-16-B2k.bel
Theory-4-5-9-16-Bk.bel
Theory-4-5-9-B1k.bel
Theory-4-5-9-B2k.bel
Theory-4-5-9-Bk.bel
Theory-4-5-B1k.bel
Theory-4-5-B2k.bel
Theory-4-5-Bk.bel
Theory-5-9-16-25-81-B1k.bel
Theory-5-9-16-25-81-B2k.bel
Theory-5-9-16-25-81-Bk.bel
Theory-5-9-16-25-B1k.bel
Theory-5-9-16-25-B2k.bel
Theory-5-9-16-25-Bk.bel
Theory-5-9-16-B1k.bel
Theory-5-9-16-B2k.bel
Theory-5-9-16-Bk.bel
Theory-5-9-B1k.bel
Theory-5-9-B2k.bel
Theory-5-9-Bk.bel
Theory-81-256-B1k.bel
Theory-81-256-B2k.bel
Theory-81-256-Bk.bel
Theory-9-16-25-81-B1k.bel
Theory-9-16-25-81-B2k.bel
Theory-9-16-25-81-Bk.bel
Theory-9-16-25-B1k.bel
Theory-9-16-25-B2k.bel
Theory-9-16-25-Bk.bel
Theory-9-16-B1k.bel
Theory-9-16-B2k.bel
Theory-9-16-Bk.bel
U1a.bel
V1r.bel
V2a.bel
"

GRAPH_DIR=$1
EXPERIMENT_DIR=`date +%Y%m%d-%H%M%S`_dyn_`hostname`

echo $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

src/pangolin-version >> $EXPERIMENT_DIR/version.txt

FLAGS="-g 0 --prefetch-async --debug -n 5"

src/benchmark-dyn $FLAGS --header >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
for g in $REMAIN; do
    g=$GRAPH_DIR/$g
    for sb in 2; do
        echo $g $sb
        src/benchmark-dyn $FLAGS --scale-binary $sb $g >> $EXPERIMENT_DIR/run.csv 2>>$EXPERIMENT_DIR/run.log
    done
done

