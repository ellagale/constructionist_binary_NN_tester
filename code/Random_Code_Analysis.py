#!/usr/bin/python
import os
import getopt
import sys
import bz2
import subprocess
import numpy as np
import develop as d

import coding_theory as code
from walker_updated import isNumber, parseDir
from coding_theory import make_prototyped_random_codes #make_prototyped_random_codes
import scipy.io as sio


########################################################################################################################
#   Randdom Code Analysis: code to test the random code NNs for generalisation and calc some stats
#
########################################################################################################################

noName = 1
includeTop = 1 #0#1#0 # don;t include the directiory you call walker ing  (avoids duplicates)
outSize = 0 # call this frm the command line to override and do only a set number of selectivity tests
verbose = 0
doAnalysis = False
doLog = True # code will use compressed log first thne fall back to log file (for output), so use this to force ti to use one or the other
doCompressedLog = True
#autostop, hack to avoid counting local codes inthe output layer
autoStop=True
categories = True
beParanoid = True # do we really want to double check all teh code?

# first parse any command line parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:i:o:v:a:",["name=","includeTop=", "outSize=", "verbose=", "doAnalysis="])
except getopt.GetoptError:
    print("{0}: [-n|--name=<name>] [-i|--includeTop=<include top level dir?>] [-o|--outSize=<output layer size>] [-v|--verbose=<be verbose?>] [-a|--doAnalysis=<do analysis?>]".format(sys.argv[0]))
    sys.exit(1)

for opt,arg in opts:
    if opt in ('-n', '--name'):
        noName = 0
        name = arg
        #print("{}".format(name))
    elif opt in ('-i', '--includeTop'):
        includeTop = int(arg)
    elif opt in ('-o', '--outSize'):
        outSize = int(arg)
    elif opt in ('-v', '--verbose'):
        verbose = int(arg)
    elif opt in ('-a', '--doAnalysis'):
        doAnalysis = True

dirlist = [x[0] for x in os.walk(os.getcwd())]
print(dirlist)
if includeTop != 1:
    dirlist = dirlist[1:]

# things we will fill in
results = {}
hamresults = {}
corr90results = {}
corr50results = {}
uncompressedFilename = 'output.log'
outputFilename = 'output.log.bz2'
analysisFilename = 'analysis.log'
select = 'Selecitivy'
noOfTestData = 1000

########################################################################################################################
# functions
########################################################################################################################

# def isNumber(n):
#     try:
#         p = float(n)
#     except ValueError:
#         return False
#     return True



def parseDir(directory):
    """Function to parse a directory and read in informations from the output file"""
    if verbose:
        print(directory)
    params=None
    ham=None
    correct=None
    bzFile = os.path.join(directory,outputFilename)
    unFile = os.path.join(directory,uncompressedFilename)
    anFile = os.path.join(directory,analysisFilename)
    cwd=os.getcwd()
    doingBz = False
#
    outputFile = None
    if os.path.isfile(bzFile) and doCompressedLog:
        outputFile = bz2.BZ2File(bzFile, 'r')
        doingBz=True
    elif params == None and os.path.isfile(unFile) and doLog:
        outputFile = open(unFile, 'r')
    if outputFile:
        corr90 = None
        corr50 = None
        isOldFormat = 1
        for rawline in outputFile:
            if doingBz:
               line =  str(rawline,'utf-8')
            else:
                line = rawline
            if line.startswith('Running with'):
                params = line.replace(',', '').split(None)
                isOldFormat = 0
                print(params)
                HLN = params[3]
            if isOldFormat and line.startswith('Layer 0 has'):
                # hack to deal with older file format
                # print('old style file')
                if verbose == 1:
                    print(line)
                params = line.replace(' ', ' ').split(None)
                params = ['m', 'e', 'h', params[3], params[1], params[6]]
            # #break
            if line.startswith('Input min Hamming'):
                ham = line.replace(':', '').split(None)[-1]
                print(line)
                # break
            if line.startswith('Training data:'):
                if corr90 == None:
                    corr90 = line.replace(':', '').split(None)[4]
                else:
                    corr50 = line.replace(':', '').split(None)[4]
                print(line)
                break
        outputFile.close()
                        # break
    if outSize == 0:
        HLN = params[3]
        if verbose == 1:
            print(params)
            print(HLN)
            print(correct)
    else:
        HLN = str(outSize)
        if verbose == 1:
            print('HLN={}'.format(HLN))
    if params==None:
        print('unable to find params for directory {}'.format(directory))
        return
    paramLine='p'.join([n.replace('.','p') for n in params if isNumber(n)])
#    with open('analysis.log','w') as analysis_file:
 #       runner = subprocess.Popen(args="~/neuralNetworks/code/NN_analysis_script1.py -n Random -H {}".format(HLN),
  ##      stdout=analysis_file,
    #    stderr=subprocess.STDOUT,
     #   cwd = os.path.join(os.getcwd(),directory))
     #   return_code = runner.wait()
      #  if return_code != 0:
       #     # print some error message here!
        #    print('error')
    if not os.path.isfile(anFile) or doAnalysis:
        print(paramLine)
        os.chdir(directory)
        print(directory)
        os.system('pwd')
        subprocess.call(['~/neuralNetworks/code/NN_analysis_script1.py', '-n', 'Random', '-H', str(HLN), '2>&1', '|', 'tee', 'analysis.log'])
#        os.system("~/neuralNetworks/code/NN_analysis_script1.py -n Random -H " + HLN + "2>&1 | tee analysis.log")
        os.chdir(cwd)
#
    count=0
    if autoStop:
        layerStop=int(HLN)/100
        if verbose:
           print('auto-stopping at layer {}'.format(layerStop))
    if os.path.isfile(anFile):
        with open(os.path.join(directory,analysisFilename),'r') as aFile:
            for line in aFile:
                if line.startswith(select):
                    count = count +1
                if line.startswith('currently on neuron:'+str(layerStop)+'p'):
                    break
        try:
            results[paramLine].append(count)
            hamresults[paramLine].append(ham)
            corr90results[paramLine].append(corr90)
            corr50results[paramLine].append(corr50)
        except KeyError:
            results[paramLine] = [ count ]
            hamresults[paramLine] = [ ham ]
            corr90results[paramLine] = [ corr90 ]
            corr50results[paramLine] = [ corr50 ]
#
    return paramLine,count

def hamming_distance(a, b):
    distance = 0
    for i in range(len(a)):
        if a[i]!=b[i]:
            distance += 1
    return distance

def all_hamming_distances(code):
    minHammingDistance = len(code[0])
    distances = []
    for a in code:
        for b in code:
            if (a != b).any():
                distances.append(hamming_distance(a, b))
    return distances



def min_hamming_distance(code):
    minHammingDistance = len(code[0])
    for a in code:
        for b in code:
            if (a != b).any():
                tmp = hamming_distance(a, b)
                if tmp < minHammingDistance:
                    minHammingDistance = tmp
    return minHammingDistance

def analyses_classes(X, noOfTrainData, noOfExamples):
    train_T_indices = [x for x in range(0, noOfTrainData, noOfExamples)]
    for i in range(0, len(train_T_indices)):
        print('i = {}'.format(i))
        if i == 0:
            this_class=X[0:train_T_indices[1]]
        else:
            this_class=X[train_T_indices[i - 1]:train_T_indices[i]]
        code.min_hamming_distance(this_class)
        distances = all_hamming_distances(this_class)
        print('Mean Hamming distance for class {} is {}'.format(i,np.mean(distances)))
        print('Std of Hamming distance for class {} is {}'.format(i, np.std(sum(this_class))))
        print('Mean no. of activiations per class {}'.format(np.mean(sum(this_class))))
        print('Example vector weight per class {}'.format(sum(this_class[0])))
    return

def test_code(noOfTestData, lenOfInput, p, weight, k, decay, verbose):
    # noOfTestData = 20
    noOfExamples = noOfTestData // noOfPrototypes
    print('{} examples per class'.format(noOfExamples))
    Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=p, weight=weight,
                                               k=k, symbolList=None, verbose=verbose, decay_templates=decay)
    analyses_classes(X=Test_X, noOfTrainData=len(Test_X), noOfExamples=noOfExamples)
    return
########################################################################################################################
#   things we read in from the output.log file
#########################################################################################################################



for directory in dirlist:
    parseDir(directory)
    #print('{0}:{1}'.format(directory,parseDir(directory)))


#We have 10 prototypes for 500 codes, giving 50 examples of each
#Prototypes have 10 blocks of length 30 each
#decaying prototype by 0.0%
decay=0.0
noOfPrototypes = 10
noOfCategories=noOfPrototypes



########################################################################################################################
#   loading the model
########################################################################################################################


def GeneralisationTest(noOfTestData=500, doPredictions=1,  doMatLabResults=False):
    """Function to create a disjoing from the training set test set"""
    X= np.load("allInputDataCurrent.npy")
    T= np.load("allOutputDataCurrent.npy")
    from keras.models import load_model
    model = load_model("Random_model.h5")


    # things we can calc from this:
    noOfTrainData = len(X)
    assert len(X) == len(T)
    lenOfInput = len(X[3])
    lenOfOutput = len(T[3])
    lenOfBlock = int(lenOfInput / noOfPrototypes)
    noOfExamples = noOfTrainData //noOfPrototypes
    noOfNewExamples = noOfTestData // noOfPrototypes
    lenOfR = lenOfInput - lenOfBlock
    weightOfX = int(sum(X[0]))
    weightOfR = weightOfX - lenOfBlock
    inverseWeightOfR = lenOfR - weightOfR
    denom=lenOfInput-(lenOfInput/noOfPrototypes) # denom is the floating point length of R
    assert int(denom) == lenOfR
    fractionalWeightOfR = weightOfR / denom
    fractionalInverseWeightOfR = inverseWeightOfR / denom
    weight = [fractionalWeightOfR, fractionalInverseWeightOfR]
    weightOfT = int(sum(T[3]))

    if lenOfOutput == noOfPrototypes:
        use1HOT = 1
    else:
        use1HOT = 0

    if categories == True:
        noOfOutputs = noOfPrototypes
        if use1HOT == 1:
            sizeOfOutput = noOfPrototypes
            print('Overwriting output vector size to length {}'.format(noOfPrototypes))
    else:
        noOfOutputs = noOfTrainData

    print('Random vector, R, has weight {0}'.format(weightOfR))

    #Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=[fractionalWeightOfR],
     #                                     k=2, symbolList=None, verbose=verbose, decay_templates=decay)


    #### testing code
    #this gives you matlab files of the codes so you can play with them if you want
    if doMatLabResults:
        Test_X = code.make_prototyped_random_codes(M=500, n=lenOfInput, p=noOfPrototypes, weight=[fractionalWeightOfR],
                                             k=2, symbolList=None, verbose=verbose, decay_templates=decay)
        sio.savemat('Test_X5000.mat', {'Test_X':Test_X})
        R = code.make_random_codes(M=500, n=501, weight=weight, k=2,symbolList=[1,0], verbose=True)
        sio.savemat('R3.mat', {'R':R})
    #######

    Test_X, All_X = code.get_test_x(X=X, noOfTestData=noOfTestData, lenOfInput=lenOfInput, noOfPrototypes=noOfPrototypes,
               weight=[fractionalWeightOfR, fractionalInverseWeightOfR], k=2, symbolList=None, verbose=verbose, decay_templates=decay)

    ###### get T
    ######
    ##  Now we get the correct sized Test_T
    Test_T, prototypeOutputCodes = code.get_test_t(T,
                                                   noOfPrototypes=noOfPrototypes,
                                                   noOfTestData=noOfTestData,
                                                   lenOfOutput=len(T[0]),
                                                   verbose=False)



    ## This sections figures out how good the model is by getting it to predict the answers for the train
    ## and test sets
    if doPredictions == 1:
        d.prediction_tester(model, X, T, name='Training data')
        if noOfTestData != 0:
            d.prediction_tester(model, Test_X, Test_T, name='Test data', example_no=0)

    np.save("GeneralisantionInputDataTest.npy", Test_X)
    np.save("GeneralisationOutputDataTest.npy", Test_T)

    return Test_X, Test_T
### now find hamming distances per class:
#test_T_indices = [x for x in range(0, noOfTestData, noOfNewExamples)]
#train_T_indices = [x for x in range(0, noOfTrainData, noOfExamples)]

_=GeneralisationTest(noOfTestData=500, doPredictions=1,  doMatLabResults=False)

exit(1)
#
# #distances = all_hamming_distances(this_class,prototypeOutputCodes[5])
#
# analyses_classes(X=Test_X, noOfTrainData=noOfTestData, noOfExamples=noOfNewExamples)
# analyses_classes(X=X, noOfTrainData=noOfTrainData, noOfExamples=noOfExamples)
#
# analyses_classes(X=X, noOfTrainData=noOfTrainData, noOfExamples=noOfExamples)
#
# i=4
# ####### more tests
#
# # nx = noOfTestData
# # lx = lenOfInput
# # np = nPrototypes
# # w(R) --> fractional weight of R
# # decay
# nx = 500
# lx = 500
# nP = 10
# fractionalWeightOfR = 1/2.
# test_code(noOfTestData=nx, lenOfInput=lx, p=nP, weight=[fractionalWeightOfR], k=2, decay=0.0, verbose=True)
#
#
#
# test_code(noOfTestData=500, lenOfInput=250, p=10, weight=[fractionalWeightOfR], k=2, decay=0.0, verbose=True)
#
#
#
# ######
# for i in range(1, len(train_T_indices)):
#     print(i)
#     code.min_hamming_distance(X[train_T_indices[i-1]:train_T_indices[i]])
#     distances=all_hamming_distances(X[train_T_indices[i-1]:train_T_indices[i]])
#     print(np.mean(distances))
#
#
#
#
#
#
# verbose=True
# #!!! THIS IS ALL FUCKED!
# P = code.make_prototype_codes(M=noOfPrototypes, n=lenOfInput, setting=1, k=2,symbolList=[1,0], verbose=verbose)
# newN = int(lenOfInput - lenOfBlock)
# R = np.zeros([noOfTrainData, newN])
# p=noOfPrototypes
# n=lenOfInput
#
#
# #R = make_random_codes(2, 500, weight=[25/50.], k=2,symbolList=[1,0], verbose=True)
#
# R = code.make_random_codes(2, 500, weight=weight, k=2,symbolList=[1,0], verbose=True)
#
# n = 0
# for p in range(noOfPrototypes):
#     for z in range(noOfExamples):
#         mask = P[p] == 0.
#         R[n,:] = X[n][mask]
#         n = n + 1 # n is the number of codewords
#
# denom=lenOfInput-(lenOfInput/noOfPrototypes)
# weightOfR = (1/3*denom)/denom
#
# Y=code.make_random_codes(M=noOfTrainData, n=newN, X=G, weight=[inverseWeightOfR], k=2,symbolList=None, verbose=False)
#
# inverseWeightOfR = 2/3
# Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=[weightOfR],
#                                       k=2, symbolList=None, verbose=True, decay_templates=decay)
# #X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[50/450.,400/450.], k=2,symbolList=None, verbose=True, decay_templates=0.2)
# # this gives a lot of local codes!
# #X = make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[210/630.,420/630.], k=2,symbolList=None, verbose=True)
# # for 700 long vectors
#
# # ella feels like being paranoid
# if beParanoid:
#     All_X = np.zeros([noOfTrainData+noOfTestData,lenOfInput])
#     All_X[range(noOfTrainData),:] = X
#     All_X[range(noOfTrainData, noOfTrainData+noOfTestData), :] = Test_X
#     duplicate_list=check_duplicate_codewords(All_X)
#     if not duplicate_list == []:
#         print('Error! Duplicates found')
#         Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=[weightOfR],
#                                       k=2, symbolList=None, verbose=True, decay_templates=decay)
#
#
#
# All_X = combine_train_test_set(X, Test_X)
#
#
# duplicate_list = code.check_duplicate_codewords(All_X)
#
# weight=[weightOfR]
#
# code.get_test_x(X, noOfTestData, lenOfInput, noOfPrototypes, weight, k=2, symbolList=None, verbose=verbose, decay_templates=decay)
#
# get_test_x(X=X, noOfTestData=100000, lenOfInput=lenOfInput, noOfPrototypes=noOfPrototypes, weight=weight,
#            k=2, symbolList=None, verbose=verbose, decay_templates=decay)
#
#
#
# Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=weight,
#                                       k=k, symbolList=symbolList, verbose=verbose, decay_templates=decay)
#
# Y = code.make_prototyped_random_codes(M=4, n=4, p=2, weight=[1/3.,2/3.],
#                                       k=2, symbolList=None, verbose=verbose, decay_templates=decay)
#
#
#
#
#
#
#
#
# ########################################################################################################################
# #   Question 1: has the model learned to generalise
########################################################################################################################