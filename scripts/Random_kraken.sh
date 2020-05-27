#!/bin/bash
set -x 

# runs Random_codes.py
# This expects a docker container
cp /command/neuralNetworks/code/Random_Code_Tester.py .

#python3 ./Random_Code_Tester.py -n Random -H $HLN -D $decay -T $traindata -e -g 0.7 -d 0.9 2>&1 | tee output.log
python3 ./Random_Code_Tester.py -n Random -H $HLN -T $traindata -e 2>&1 | tee output.log
#python3 ./Random_Code_Tester.py -n Random -H $HLN -D $decay -T $traindata -g $std -e 2>&1 | tee output.log
#cp ../../code/Random_Code_Tester_1205.py .
cp /command/neuralNetworks/code/Run_kmeans.py .
python ./Run_kmeans.py -n Random 2>&1 | tee kMeans.log
cp /command/neuralNetworks/code/NN_analysis_script1.py .
python ./NN_analysis_script1.py -n Random -H $HLN 2>&1 | tee analysis.log
OF=$(date -d "today" +"%Y%m%d_%H%M")
mkdir $OF
cp *.* $OF
rm *.npz
rm *.png
