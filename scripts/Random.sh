#!/bin/bash
set -x 
#export HLN=800
# runs Random_codes.py 
# copy codes here!
cp code/Random_Code_Tester.py .

python3 ./Random_Code_Tester.py -n Random -H $HLN -D $decay -T $traindata -g 0.7 -e 2>&1 | tee output.log
#cp ../../code/Random_Code_Tester_1205.py .
cp /code/Run_kmeans.py .
python ./Run_kmeans.py -n Random 2>&1 | tee kMeans.log
cp code/NN_analysis_script1.py .
python ./NN_analysis_script1.py -n Random -H $HLN 2>&1 | tee analysis.log
OF=$(date -d "today" +"%Y%m%d_%H%M")
mkdir $OF
cp *.* $OF
rm *.npz
rm *.png
