#!/bin/sh

export HLN=800
export traindata=500
export decay=0.0

# this is run inside a docker container (Kraken was the name of my machine with a docker environment)
/command/neural_networks/scripts/Random_kraken.sh
