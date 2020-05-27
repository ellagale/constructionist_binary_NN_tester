#!/bin/bash

# use this to run over directories (make them first)

for dirname in dir_a dir_b dir_c
do
    cd $dirname
    #cp ~/neuralnetworks/scripts/Test.sh .
    #./Test.sh
    pwd
    ./runbatch.sh
    cd ..
done
