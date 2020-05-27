#!/usr/bin/python
import os
import walker
import develop as d
import numpy as np
import coding_theory as code
import numbers

##############################################################################
#   set- up values
##############################################################################


noOfTrainData = 500
sizeOfOutput = 50 # this is the size of the output vectors
lenOfInput = 500
inSize=lenOfInput
noOfTestData = 0 # THIS EXPERIMENT USES NO TEST DATAS!!!
doKmeans =0
#noOfEpochs= 45000
modelValidationDataRatio=0
noOfLayers=1
#HLN = 800
decay = 0.0
name = 'Random'
verbose = True
categories = True
noOfPrototypes = 10
noOfExamples = noOfTrainData/noOfPrototypes
# set categories to True for many->one mapping, False for many -> many mapping
if categories == True:
    noOfOutputs = noOfExamples
else:
    noOfOutputs = noOfTrainData



############################################################################
#   getting directory and selectivity codes taken from walker.py
############################################################################

dirlist=[x[0] for x in os.walk(os.getcwd())]

results = {} 

outputFilename='output.log'
analysisFilename='analysis.log'
select='Selecitivy'



for directory in dirlist:
    walker.parseDir(directory)
    print('{0}:{1}'.format(directory,walker.parseDir(directory)))

for key in results:
    print 'p{0} = {{{1} }};'.format(key,','.join(map(str,results[key])))


###########################################################################
### read shit in taken from NN_analysis_script
############################################################################

# TODO: sensibly read this data in from something created by develop or at least get the data names
# this gets our data
noName=0
name = "Random"
if noName == 1:
    model = d.ModelBuilder.load("model.h5")

    netStats = d.NetworkStats.load('stats.h5')
else:
    model = d.ModelBuilder.load(name+"_model.h5")

    netStats = d.NetworkStats.load(name + '_stats.h5')

noOfLayers = len(netStats.stats)
noOfNeurons = [netStats.stats[i].shape[1] for i in range(noOfLayers)]

# use this when you're not testing all the dks
# this assumes you want to plot everything, but only have testSet kmeans!
# use this for preliminary data as kmeans is expensive
testSet = [range(8),[],[]]


allInputData = np.load("allInputDataCurrent.npy")
allOutputData = np.load("allOutputDataCurrent.npy")

#############################################################################
#   hamming distance code takn from Random_Code_Tester
#############################################################################

t=allOutputData
X = allInputData
XTrain = X


temp = code.min_hamming_distance(t)
print('Output code: This is a [{0}, {1}, {2}] code'.format(noOfOutputs, sizeOfOutput, temp))
print("Outputmin Hamming distance: %i" % temp)
weights = code.weight(t, verbose=True)
temp2 = code.min_hamming_distance(XTrain)
print('Input code: This is a [{0}, {1}, {2}] code'.format(noOfTrainData, lenOfInput, temp2))
print("Input min Hamming distance: %i" % temp2)