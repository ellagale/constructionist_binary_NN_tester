#!/usr/bin/python
import os
import getopt
import sys
import bz2
import subprocess
import numpy as np
import coding_theory as code
import develop as d
import scipy.io as sio



# set defaults
noName = 1
includeTop = 0 # don;t include the directiory you call walker ing  (avoids duplicates)
outSize = 0 # call this frm the command line to override and do only a set number of selectivity tests
verbose = 0
doAnalysis = False
doLog = True # code will use compressed log first thne fall back to log file (for output), so use this to force ti to use one or the other
doCompressedLog = True 
#autostop, hack to avoid counting local codes inthe output layer
autoStop=True
doGeneralisation = False
noOfTestData = 500

# first parse any command line parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:i:o:v:a:g:t:",
                               ["name=","includeTop=", "outSize=", "verbose=", "doAnalysis=", "doGeneralisation", "noOfTestData"])
except getopt.GetoptError as e:
    print(e)
    print("{0}: [-n|--name=<name>] [-i|--includeTop=<include top level dir?>] [-o|--outSize=<output layer size>] [-v|--verbose=<be verbose?>] [-a|--doAnalysis=<do analysis?>] [-g|--doGeneralisation=<do generalisation>] [-t=use int test ]".format(sys.argv[0]))
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
    elif opt in ('-g', '--doGeneralisation'):
        doGeneralisation = True
    elif opt in ('-t', '--doGeneralisationChangeTest'):
        doGeneralisation = True
        noOfTestData = int(arg)


dirlist = [x[0] for x in os.walk(os.getcwd())]
print(dirlist)
if includeTop != 1:
    dirlist = dirlist[1:]

results = {} 
hamresults = {}
corr90results = {}
corr50results = {}
gencorr90results = {}
gencorr50results = {}
uncompressedFilename='output.log'
outputFilename='output.log.bz2'
analysisFilename='analysis.log'
generalisationFilename='generalisation.log'
select='Selecitivy'

print('This file is walker_updated which reads accuracies from output.log or output.bz2')
print('walker_updated will be faster if you do not want the accuracies')

##### functions

def isNumber(n):
    try:
        p = float(n)
    except ValueError:
        return False
    return True


def GeneralisationTest(noOfTestData=500, noOfPrototypes = 10, decay=0.0, doPredictions=1,  doMatLabResults=False):
    """Function to create a disjoint from the training set test set"""
    categories = True
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

def parseDir(directory):
    if verbose:
        print(directory)
    params=None
    ham=None
    correct=None
    bzFile = os.path.join(directory,outputFilename)
    unFile = os.path.join(directory,uncompressedFilename)
    anFile = os.path.join(directory,analysisFilename)
    gnFile = os.path.join(directory,generalisationFilename)
    cwd=os.getcwd()

        #
    if os.path.isfile(bzFile) and doCompressedLog:
        noOfPrototypes = None
        decay = None
        with bz2.BZ2File(bzFile, 'r') as oFile:
            corr90 = None
            corr50 = None
            for bLine in oFile:
                line = bLine.decode("utf-8")
                
                if line.startswith('Running with'):
                    params=line.replace(',','').split(None)
                    print(params)
                    HLN=params[3]
                    continue
#                    #break
                if line.startswith('Input min Hamming'):
                    ham=line.replace(':' , '').split(None)[-1]
                    print(line)
                    continue
                if line.startswith('We have '):
                    blocks = line.split(None)
                    if blocks[3] == 'prototypes':
                        noOfPrototypes = int(blocks[2])
                        print(line)
                        continue
                if line.startswith('decaying prototype by'):
                    decay = float(line.replace(':', '').split(None)[-1][:-1])
                    print(line)
                    continue
                    #break


                if line.startswith('Training data:'):
                    if corr90 == None:
                        corr90 = line.replace(':' , '').split(None)[4]
                    else:
                        corr50 = line.replace(':' , '').split(None)[4]
                    print(line)
                    continue
                    # break?
    if params==None and os.path.isfile(unFile) and doLog:
        with open(os.path.join(directory,uncompressedFilename),'r') as oFile:
            corr50 = None
            corr90 = None
            for line in oFile:
                if line.startswith('Running with'):
                    params=line.replace(',','').split(None)
                    if verbose == 1:
                        print(line)
#                     #break
                if line.startswith('Input min Hamming'):
                    ham=line.replace(':' , '').split(None)[-1]
                    if verbose:
                        print(line)
                    #break
                if line.startswith('We have '):
                    blocks = line.split(None)
                    if blocks[3] == 'prototypes':
                        noOfPrototypes = int(blocks[2])
                        print(line)
                        continue
                if line.startswith('decaying prototype by'):
                    decay = float(line.replace(':', '').split(None)[-1][:-1])
                    print(line)
                    continue
                            #
                # if line.startswith ('Layer 0 has'):
                #     # hack to deal with older file format
                #     #print('old style file')
                #     if verbose == 1:
                #         print(line)
                #     params = line.replace(' ', ' ').split(None)
                #     params = ['m','e','h',params[3], params[1], params[6]]
                    #print(params)
                    #break
                #if line.startswith('Training data:'):
                 #   if corr50 == None:
                 #       corr50 = line.replace(':' , '').split(None)[4]
                 #   else:
                 #       corr90 = line.replace(':' , '').split(None)[4]
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

    count=0
    if autoStop:
        if int(HLN)> 100:
            layerStop=int(HLN)/100
        else:
            layerStop=int(HLN)
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
            #corr90results[paramLine].append(corr90)
            #corr50results[paramLine].append(corr50)
        except KeyError:
            results[paramLine] = [ count ]
            hamresults[paramLine] = [ ham ]
            #corr90results[paramLine] = [ corr90 ]
            #corr50results[paramLine] = [ corr50 ]
    if doGeneralisation:
        os.chdir(directory)
        sys.stdout=open(generalisationFilename, "w")
        GeneralisationTest(noOfTestData=500, noOfPrototypes=noOfPrototypes, decay=decay, doPredictions=1,
                           doMatLabResults=False)
        sys.stdout=sys.__stdout__
        os.chdir(cwd)
    if os.path.isfile(gnFile):
        gencorr50 = None
        gencorr90 = None
        with open(os.path.join(directory,generalisationFilename),'r') as gFile:
            for line in gFile:
                if line.startswith('Training data:'):
                    if corr50 == None:
                        corr50 = line.replace(':' , '').split(None)[4]
                    else:
                        corr90 = line.replace(':' , '').split(None)[4]
                if line.startswith('Test data:'):
                    if gencorr50 == None:
                        gencorr50 = line.replace(':' , '').split(None)[4]
                    else:
                        gencorr90 = line.replace(':' , '').split(None)[4]
        try:
            gencorr90results[paramLine].append(gencorr90)
            gencorr50results[paramLine].append(gencorr50)
            corr90results[paramLine].append(corr90)
            corr50results[paramLine].append(corr50)
        except KeyError:
            gencorr90results[paramLine] = [ gencorr90 ]
            gencorr50results[paramLine] = [ gencorr50 ]
            corr90results[paramLine] = [ corr90 ]
            corr50results[paramLine] = [ corr50 ]

    return paramLine,count
    
for directory in dirlist:
    parseDir(directory)
    #print('{0}:{1}'.format(directory,parseDir(directory)))

#for key in results:
#   print 'p{0} = {{{1} }};'.format(key,','.join(map(str,results[key])))

for key in results:
    if noName == 0:
        print('{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, results[key])),name))
        print('Ham{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, hamresults[key])) , name))
        print('AccWithin90{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, corr90results[key])) , name))
        print('AccWithin50{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, corr50results[key])) , name))
        print('GenAccWithin90{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, gencorr90results[key])) , name))
        print('GenAccWithin50{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, gencorr50results[key])) , name))
    else:
        print('p{0} = {{{1} }};'.format(key, ','.join(map(str, results[key]))))
        print('Hamp{0} = {{{1} }};'.format(key, ','.join(map(str, hamresults[key]))))
        print('AccWithin90p{0} = {{{1} }};'.format(key, ','.join(map(str, corr90results[key]))))
        print('AccWithin50p{0} = {{{1} }};'.format(key, ','.join(map(str, corr50results[key]))))
        print('GenAccWithin90p{0} = {{{1} }};'.format(key, ','.join(map(str, gencorr90results[key]))))
        print('GenAccWithin50p{0} = {{{1} }};'.format(key, ','.join(map(str, gencorr50results[key]))))



