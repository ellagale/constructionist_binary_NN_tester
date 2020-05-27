#!/bin/bash

# use this to run one experiment

for traindata in 500 #1000 250 #nx
do
    #for decay in 0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.35 0.4 0.45 0.5 
    for decay in 0.0 #0.275 0.3 0.35 0.4 0.45 0.5 
    #for decay in 0.0 
    do
        export traindata
        export decay
        #for HLN in 50 75 100 150 200 250 300 350 400 450 490 500 510 550 600 700 800 900 1000 1500 2000 3000 4000 5000 7000 10000  
        for HLN in 500   
        #for HLN in 500 1000
        do 
            export HLN
            #for run in 1 2 3 4 5 6 7 8 9 10
            for run in 1
            do 
                ./Random.sh
            done
        done
    done
done
