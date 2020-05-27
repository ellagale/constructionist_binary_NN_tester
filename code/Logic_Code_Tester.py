#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt

import keras
import keras.backend as K
from keras.preprocessing import image

import cluster_stats as cs
import develop as d
import analysis as a
import coding_theory as code
import random

"""This script is for testing arbitrary random codes"""

#####################################################
## settings
#####################################################
# standard settings
doPlots=1
showPlots=0
savePlots=1

# experiment settings
#use1HOT = 0
#scaled =1 # use downsampled as in the paper
#preprocessing = 'none'#'standard' # standard = zero mean and normalised
noOfTrainData = 576
sizeOfOutput = 8 # this is the size of the output vectors
lenOfInput = 14
inSize=lenOfInput
noOfTestData = 0 # THIS EXPERIMENT USES NO TEST DATAS!!!
doKmeans =0
noOfEpochs= 3000
modelValidationDataRatio=0
noOfLayers=1
HLN = 15
decay = 0.0
name = 'Logic'
verbose = True
categories = False

dataset = np.loadtxt(d.get_data_path('logic_puzzles', 'logic_1HOT.csv'), delimiter=",")
#dataset = np.loadtxt(d.get_data_path('logic_puzzles', 'logic_puzzles_complete.csv'), delimiter=",")
sizeOfInput = 14
sizeOfOutput = 8

#modelName = 'three_layer_1_HOT_out'
#noOfLayers = 3

# set categories to True for many->one mapping, False for many -> many mapping
if categories == True:
    noOfOutputs = noOfExamples
else:
    noOfOutputs = noOfTrainData

# read any command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:H:D:T:",["name=","HLN=","decay=","traindata="])
except getopt.GetoptError:
    print("{0}: [-n|--name=<name>] [-H|--HLN=<neuron count>] [-D|--decay=<decay probability>] [-T|--traindata=<size of training data>]".format(sys.argv[0]))
    sys.exit(1)
for opt,arg in opts:
    if opt in ('-n', '--name'):
        noName = 0
        name = arg
    elif opt in ('-H', '--HLN'):
        HLN = int(arg)
    elif opt in ('-D', '--decay'):
        decay = float(arg)
    elif opt in ('-T', '--traindata'):
        noOfTrainData = int(arg)

print('Running with HLN {0}, decay {1}, train data size {2}'.format(HLN,decay,noOfTrainData))

sizeOfValidationData = 0  # --> noOfValidData
modelValidationDataRatio = 0.3# --> validation split that the model chooses from training data
noOfTestData = 192  # --> noOfTestData

# things that rely on read in code above
if categories == True:
    noOfExamples = noOfTrainData / noOfPrototypes
    noOfOutputs = noOfExamples
else:
    noOfOutputs = noOfTrainData


######################################################
# 0. Get the codes!
######################################################

## this makes a set of random vectors based on a perpendicular basis prototype vectors
#X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[150/450.,300/450.],
    #                                  k=2,symbolList=None, verbose=True, decay_templates=decay)
#X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[50/450.,400/450.], k=2,symbolList=None, verbose=True, decay_templates=0.2)
# this gives a lot of local codes!
#X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[210/630.,420/630.], k=2,symbolList=None, verbose=True)
# for 700 long vectors

## this makes our output vectors - relies on C being sorted (which it is)
#T = np.zeros([noOfTrainData, sizeOfOutput])
#t = code.make_random_codes(M=noOfOutputs, n=sizeOfOutput, weight=[0.5], k=2, verbose=True)
#n = 0
#for p in range(noOfPrototypes):
#    for z in range(noOfExamples):
#        T[n, :] = t[p]
#        n = n + 1

# at the moment, we are not doing validation:
#XTrain = X
#TTrain = T
#allInputData=X


XTrain, TTrain, XTest, TTest = d.dataset_slicer(dataset, sizeOfInput, sizeOfOutput,
                                                shuffle=True, sizeOfTestingData=noOfTestData,
                                                sizeOfValidationData = sizeOfValidationData,
                                                modelValidationDataRatio=modelValidationDataRatio)

allInputData=np.append(XTrain, XTest, axis=0)
allOutputData=np.append(TTrain, TTest, axis=0)

np.save("allInputDataCurrent", allInputData)
np.save("allOutputDataCurrent", allOutputData)

# XAll and TAll, only use this for analysing a trained NN
#XAll, TAll = d.dataset_slicer(dataset, sizeOfInput, sizeOfOutput,
#                                                shuffle=True, sizeOfTestingData=0,
#                                                sizeOfValidationData = 0,
 #                                               modelValidationDataRatio=0)

#model = d.build_model(sizeOfInput, sizeOfOutput)
inSize = sizeOfInput
outSize = sizeOfOutput
t=TTrain
X=XTrain
T=TTrain

#############################################################
# get some input stats
#############################################################

if verbose == True:
#    temp=code.min_hamming_distance(T)
#    print('Input code: This is a [{0}, {1}, {2}] code'.format(noOfInputs, sizeOfInput, temp))
#    print("min Hamming distance: %i" % temp)
#    weights = code.weight(T, verbose=True)
    temp=code.min_hamming_distance(t)
    print('Output code: This is a [{0}, {1}, {2}] code'.format(noOfOutputs, sizeOfOutput, temp))
    print("Outputmin Hamming distance: %i" % temp)
    weights = code.weight(t, verbose=True)
    temp2 = code.min_hamming_distance(XTrain)
    print('Input code: This is a [{0}, {1}, {2}] code'.format(noOfTrainData, lenOfInput, temp2))
    print("Input min Hamming distance: %i" % temp2)


# N.B. for the analysis scripts
np.save("allInputDataCurrent", X)
np.save("allOutputDataCurrent", T)

# N.B. we are just using this to get intel on the input
dataset = np.append(X, T, axis=1)
d.dataset_slicer(dataset, sizeOfInput=len(X[0, :]), sizeOfOutput=sizeOfOutput, sizeOfTestingData=0, shuffle=True,
                     sizeOfValidationData=0, modelValidationDataRatio=0)

#################################################################
# 1. Build the model
#################################################################

#K.set_learning_phase(0)

model = d.ModelBuilder.add_init_layer(inSize, activationFunction='sigmoid', layerSize=HLN) \
    .add_output_layer(outputSize=sizeOfOutput, activationFunction='sigmoid')

#    .add_dense_layer(activationFunction='sigmoid', ) \

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy', 'cosine_proximity'])

#callbacks = [StopMaxedAccuracy(monitor='acc', value=1.0, mode='max')]

##########################################################################
# callbacks to monitor the minimsation process
#########################################################################

activationCallback = d.activation_history()

## this thing csv_logger writes the mtrics logged out to a csv file
csv_logger = keras.callbacks.CSVLogger('training.log')

## checkpoints weights during training
#checkpointer = keras.callbacks.ModelCheckpoint(filepath="./tmp/weights.hdf5", verbose=1, save_best_only=True)

## early stoppping monitor
early_stopper=keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=100, verbose=1, mode='auto')

##################################################################
# 2. Train the model
##################################################################

fit = model.fit(X, T, nb_epoch=noOfEpochs, batch_size=32, validation_split=modelValidationDataRatio,
                shuffle=True, callbacks=[activationCallback, csv_logger, early_stopper])

##################################################################
# 3. Test the model
##################################################################

metrics = model.evaluate(XTrain, TTrain, batch_size=32)
if noOfTestData != 0:
    metrics2 = model.evaluate(XTest, TTest, batch_size=32)

## Output datas
weights = []
i = 0
for layer in model.layers:
    weights.append(layer.get_weights())
    i = i + 1

# model.train_on_batch(data, labels)

# loss_and_metrics = model.evaluate(data, labels, batch_size=32)
print('Training and validation accuracies')
print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1] * 100))
if noOfTestData != 0:
    print("%s: %.2f%%" % (model.metrics_names[1], metrics2[1] * 100))
#print("%s: %.2f%%" % (model.metrics_names[2], metrics[2] * 100))
#print("%s: %.2f%%" % (model.metrics_names[2], metrics2[2] * 100))
#print("%s: %.2f%%" % (model.metrics_names[3], metrics[3] * 100))
#print("%s: %.2f%%" % (model.metrics_names[3], metrics2[3] * 100))

# this should be an integer, if it is not, something has gone horribly wrong!
print("\n%s: %i" % ('No Of mistakes in training data: ', noOfTrainData * (1 - metrics[1])))
if noOfTestData != 0:
    print("%s: %i" % ('No Of mistakes in test data: ', noOfTestData * (1 - metrics2[1])))
print('')
# classes = model.predict_classes(TTest, batch_size=32)
# proba = model.predict_proba(data, batch_size=10)
# model.save('my_model.h5')
#print("\n%s: %i" % ('No Of mistakes in training data: ',
#                    100*(d.noOfTrainData * (1 - metrics[1]))/d.noOfTrainData))
#print("%s: %i" % ('No Of mistakes in test data: ',
#                  100*(noOfTestData * (1 - metrics2[1])))/noOfTestData)
print('')

# with a Sequential model

#################################################################
# !! Here we get the activations of hte model on all the data !!#
#################################################################


netStats = d.NetworkStats(model, allInputData)

out = netStats.stats

for i in range(noOfLayers):
    # print(out[i])
    r, c = out[i].shape
    print('Layer {0} has {1} neurons and {2} data'.format(i, c, r))
    r = out[i][:, 2].shape[0]
    # print('layer {0} has {1} LC'.format(i, LocalCharacter))

#import numpy as np
import random






# print(layer_output, r, c)
# !! values for normalisation need to be set above! It depends on the activation function



noOfStatsTests = 2;  ## set this up earlier if you decide to give an option about which tests to do
stats = list()
for i in range(noOfLayers):
    # in layer 2 grab the 5th column
    r, c = out[i].shape
    stats = np.zeros((noOfStatsTests, c))
    for j in range(c):
        r = out[i][:, j].shape[0]
        y = out[i][:, j]
        # stats[0, j] = a.local_charactor(y)
        # stats[1, j] = a.global_charactor(y)
        # ynorm = a.normalise_to_zero_one_interval(y, layerMinMaxVal[i][0], layerMinMaxVal[i][1])
        # print(stats)

doPredictions = 1
## This sections figures out how good the model is by getting it to predict the answers for the train
## and test sets
if doPredictions == 1:
    d.prediction_tester(model, XTrain, TTrain, name='Training data')
    if noOfTestData != 0:
        d.prediction_tester(model, XTest, TTest, name='Test data', example_no=0)




from keras.utils.visualize_util import plot
plot(model, to_file='model.png')


d.write_out(model, "model.json")


doGap = 0
if doGap ==1:
    verbose = 1
    if verbose == 1:
        print("dk.G={}".format(dk.G))
        print("dk.Wks={}".format(dk.Wks))
        print("dk.Wkbs={}".format(dk.Wkbs))
    kd = np.array(range(noOfK - 1)) + 1
    a.plotter(kd, np.array([dk.Wks[0:-1], dk.Wkbs[0:-1]]),
            legend=['observed', 'expected'], labels=['log($W_k$)', 'No. of clusters, K'])
    kd = np.array(range(noOfK-1)) + 1
    a.plotter(kd, np.array([dk.G]))

    estimates=cs.K_estimator(dk.G, flag='doGap')
    print('K (from gap stat) = {}'.format(estimates))




if doPlots != 0:
    if doPlots == 2 or doPlots == 3:
        # this plots the new spotty plotsi
        for l in range(noOfLayers):
            fig = plt.figure(l)
            t = a.jitterer(out, l)
            # yrange=max(out[l])-min(out[l])
            plt.plot(t, out[l].T, '+', label='training')
            plt.ylabel('Activation')
            plt.xlabel('Neuron no.')
            # plt.axis([min(t)-0.25, max(t)+0.25, min(out[l])-0.1*yrange, max(out[l]+0.1*yrange)])
            # plt.legend()
            if showPlots == 1:
                plt.show()
            if savePlots == 1:
                fig.savefig('temp' + str(l) + '.png', dpi=fig.dpi)

    if doPlots == 1 or doPlots == 3:
        # this plots the loss and training data
        t = np.arange(0., 1, noOfEpochs)

        fig=plt.figure(1)
        plt.plot(fit.history['loss'], 'o', label='training')
        if modelValidationDataRatio > 0:
            plt.plot(fit.history['val_loss'], 'o', label='validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        if showPlots == 1:
            plt.show()
        if savePlots == 1:
            fig.savefig('loss.png', dpi=fig.dpi)

        plt.figure(2)
        plt.plot(fit.history['acc'], '-', label='training')
        if modelValidationDataRatio > 0:
            plt.plot(fit.history['val_acc'], '-', label='validation')
        plt.ylabel('accuracy ')
        plt.xlabel('epoch')
        plt.legend()
        if showPlots == 1:
            plt.show()
        if savePlots == 1:
            fig.savefig('acc.png', dpi=fig.dpi)

        #plt.figure(3)
        #plt.plot(activationCallback.activations, '-', label='training')
        #if modelValidationDataRatio > 0:
        #    plt.plot(fit.history['val_acc'], '-', label='validation')
        #plt.ylabel('accuracy ')
        #plt.xlabel('epoch')
        #plt.legend()
        #plt.show()

    plt.savefig("figure.png")

if doKmeans == 0:
    doF =0
else:
    doF=1

if doF == 1:
    kd = np.array(range(noOfK)) + 1
    a.plotter(kd, np.array([dk.fs, 0.9*np.ones(noOfK)]),
            labels=['F(K) measure', 'No. of clusters, K'], xaxis=1.)
    estimates=cs.K_estimator(dk.fs, flag='doF')
    print('K (from f(K)) = {}'.format(estimates))


## writing model to file and reading it back in test



from keras.utils.visualize_util import plot
plot(model, to_file='{0}_model.png'.format(name))

d.write_out(model, "{0}_model.json".format(name))


netStats.save('{0}_stats.h5'.format(name))
model.save('{0}_model.h5'.format(name))


