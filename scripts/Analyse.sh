#!/bin/bash
set-c

for i in $( find . -type d) 
    do pushd "$i" 
        pwd 
        #cp ~/neuralNetworks/code/Run_kmeans.py .
        #python ./Run_kmeans.py -n Random 2>&1 | tee kMeans.log
        cp ~/neuralNetworks/code/NN_analysis_script1.py .
#        HLN=sed -n 's/Running with HLN \([0-9]*\).*/\1/p' output.log
        HLN=$(sed -n 's/Running with HLN \([0-9]*\).*/\1/p' output.log)
        python ./NN_analysis_script1.py -n Random -H $HLN 2>&1 | tee analysis.log
        popd 
    done
