#!/bin/bash
# use this the run batches of experiments

for traindata in 1000 500 100
do
    #for decay in 0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.35 0.4 0.45 0.5
    #for decay in 0.275 0.3 0.35 0.4 0.45 0.5 
    for decay in 0.0
    #for epochs in 1000 2500 5000 7500 10000 12500 15000 17500 20000 25000 30000 35000 40000 45000
    do
        export traindata
        export decay
        export epochs
        for run in 1 2 3 4 5 6 7 8 9 0 11 12 13 14 15
            do
            for HLN in 250 500
        #for HLN in 50 75 100 150 200 250 300 350 400 450 490 500 510 550 600 700 800 900 1000 1500 2000 3000 4000 5000 7000 
        #for HLN in 250 500 1000 2000
            do
            export HLN
#            ~/neuralNetworks/scripts/Random.sh
                ./Random.sh
            done
        done
    done
done
