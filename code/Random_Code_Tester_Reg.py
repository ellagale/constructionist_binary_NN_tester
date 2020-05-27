#!/usr/bin/python
from __future__ import absolute_import

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import scipy.io as sio

"""This script is for testing arbitrary random codes
Now with added regularizers"""

#####################################################
## settings
#####################################################
#with tf.device(/cpu:0)

# standard settings
doPlots=1
showPlots=0
savePlots=1

# experiment settings
use1HOT = 0 # N.b> this iverwrites output catagories
#scaled =1 # use downsampled as in the paper
#preprocessing = 'none'#'standard' # standard = zero mean and normalised
noOfTrainData = 500
sizeOfOutput = 100 # this is the size of the output vectors # ! change this if using efficient 1-HOT
lenOfInput = 500
inSize=lenOfInput
noOfTestData = 0 # THIS EXPERIMENT USES NO TEST DATAS!!!
doKmeans =0
noOfEpochs= 60000
modelValidationDataRatio=0
noOfLayers=1
HLN = 800
decay = 0.0
name = 'Random'
verbose = True
categories = True
noOfPrototypes = 10
doDropout = False
dropout = 0.2 # amount of neurons to drop out
activation_function = 'relu' #'sigmoid' #'relu'
useGPU = True
doEarlyStopper = False
doNoise = True
std = 0.0
doReg = False
# set categories to True for many->one mapping, False for many -> many mapping


# read any command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:H:D:d:T:E:F:ceg:",
                               ["name=","HLN=","decay=","dropout=","traindata=","epoch=","function=a,""cpu","earlystopper","gaussian=","regularizer="])
except getopt.GetoptError:
    print("{0}: [-n|--name=<name>] [-H|--HLN=<neuron count>] [-D|--decay=<decay probability>]"
          "[-T|--traindata=<size of training data>] [-E|--epoch=<maximum epoch>] [-F|--function=<activation function>]"
          "[-c|--cpu] [-e|--earlystopper] [-g|--gaussian] [-R|--regularizer]".format(sys.argv[0]))
    sys.exit(1)
for opt,arg in opts:
    if opt in ('-n', '--name'):
        name = arg
    elif opt in ('-H', '--HLN'):
        HLN = int(arg)
    elif opt in ('-D', '--decay'):
        decay = float(arg)
    elif opt in ('-d', '--dropout'):
        doDropout = True
        dropout = float(arg)
    elif opt in ('-T', '--traindata'):
        noOfTrainData = int(arg)
    elif opt in ('-E', '--epoch'):
        noOfEpochs = int(arg)
    elif opt in ('-c', '--cpu'):
        useGPU = False
    elif opt in ('-F', '--function'):
        activation_function = arg
    elif opt in ('-e', '--earlystopper'):
        doEarlyStopper = True
    elif opt in ('-g', '--gaussian'):
        doNoise = True
        std = float(arg)
    elif opt in ('-R', '--regularizer'):
        doReg = True
        reg = str(arg)


if doDropout:
    print('Dropout is activated at {}'.format(dropout))

if doEarlyStopper:
    print('Early stopper is activated')

if not useGPU:
     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
     os.environ["CUDA_VISIBLE_DEVICES"] = ""
     print('GPU disabled.')
if doNoise:
    print('Gaussian Noise is activated with a std of {}'.format(std))

if doReg:
    if reg == 'L1':
        print('Using L1 regularisation')

print('Running with HLN {0}, decay {1}, train data size {2} for up to {3} epochs under name {4} with noise {5}'
      .format(HLN,decay,noOfTrainData,noOfEpochs,name,std))


# things that rely on read in code above
noOfExamples = noOfTrainData//noOfPrototypes
noOfCatagories = noOfPrototypes
if categories == True:
    noOfOutputs = noOfPrototypes
    if use1HOT == 1:
        sizeOfOutput = noOfPrototypes
        print('Overwriting output vector size to length {}'.format(noOfPrototypes))
else:
    noOfOutputs = noOfTrainData

######################################################
# 0. Get the codes!
######################################################

## this makes a set of random vectors based on a perpendicular basis prototype vectors
#weightOfR = 90/270.
#inverseWeightOfR = 180/270.
#weightOfR = 150/450.
#inverseWeightOfR = 300/450.
denom=lenOfInput-(lenOfInput/noOfPrototypes)
weightOfR = (1/3*denom)/denom

inverseWeightOfR = 2/3
X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=noOfPrototypes, weight=[weightOfR],
                                      k=2,symbolList=None, verbose=True, decay_templates=decay)
#X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[50/450.,400/450.], k=2,symbolList=None, verbose=True, decay_templates=0.2)
# this gives a lot of local codes!
#X = code.make_prototyped_random_codes(M=noOfTrainData, n=lenOfInput, p=10, weight=[210/630.,420/630.], k=2,symbolList=None, verbose=True)
# for 700 long vectors


print('Random vector, R, has weight  {0}'.format(weightOfR))

## this makes our output vectors - relies on C being sorted (which it is)

if use1HOT==1:
    # we use the no of catagories and overwrite the output vectors size
    T = np.zeros([noOfTrainData, sizeOfOutput])
    t = d.make_1HOT(sizeOfOutput)
    print('You are using 1-HOT output encoding')
    print(t)
else:
    T = np.zeros([noOfTrainData, sizeOfOutput])
    t = code.make_random_codes(M=noOfOutputs, n=sizeOfOutput, weight=[0.5], k=2, verbose=True)
    print('You are using distributed output encoding')
    print(t)
n = 0
for p in range(noOfPrototypes):
    for z in range(noOfExamples):
        T[n, :] = t[p]
        n = n + 1

# at the moment, we are not doing validation:
XTrain = X
TTrain = T
allInputData=X
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

###!!!# this needs to go into develop at some point!
# -*- coding: utf-8 -*-

#from __future__ import absolute_import
#from keras.engine import Layer
#from keras import backend as K
#import numpy as np
#from keras.legacy import interfaces
#from keras.engine import InputSpec
#class GaussianNoiseForSigmoid(Layer):
#    """Apply additive 0.5-centered Gaussian noise.
#    This is useful to mitigate overfitting
#    (you could see it as a form of random data augmentation).
#    Gaussian Noise (GS) is a natural choice as corruption process
#    for real valued inputs.
#    As it is a regularization layer, it is only active at training time.
#    # Arguments
#        stddev: float, standard deviation of the noise distribution.
#    # Input shape
#        Arbitrary. Use the keyword argument `input_shape`
#        (tuple of integers, does not include the samples axis)
#        when using this layer as the first layer in a model.
#    # Output shape
#        Same shape as input.
#    """

 #   @interfaces.legacy_gaussiannoise_support
 #   def __init__(self, stddev, **kwargs):
 #       super(GaussianNoiseForSigmoid, self).__init__(**kwargs)
 #       self.supports_masking = True
 #       self.stddev = stddev
#
#    def call(self, inputs, training=None):
#        def noised():
#            return inputs + K.random_normal(shape=K.shape(inputs),
#                                            mean=0.5,
#                                            stddev=self.stddev)
#        return K.in_train_phase(noised, inputs, training=training)
#
#    def get_config(self):
#        config = {'stddev': self.stddev}
 #       base_config = super(GaussianNoiseForSigmoid, self).get_config()
 #       return dict(list(base_config.items()) + list(config.items()))

#from keras import backend as K

## example of l1 regularlizer
#def l1_reg(weight_matrix):
#    return 0.01 * K.sum(K.abs(weight_matrix))

#model.add(Dense(64, input_dim=64,
 #               kernel_regularizer=l1_reg))


import keras.backend as K
K.set_learning_phase(1)
#.add_dropout_layer(layerSize=HLN, dropout=0.2) \

doingByHand = 1
if doDropout:
    print('Using Dropout = {}!'.format(dropout))
    model = d.ModelBuilder.add_init_layer(inSize, activationFunction=activation_function, layerSize=HLN) \
                         .add_dropout_layer(dropout) \
                         .add_output_layer(outputSize=sizeOfOutput, activationFunction=activation_function)
elif doingByHand==1:
    print('Using hand hacked NN so please check Random_Code_Reg_Tester for actual NN details')
    # use this code to hack nn by hand
    from keras.layers import Dense, Dropout, Activation, GaussianNoise
    from keras.models import Model, Sequential
    from develop import Gaussian_Noise_For_Sigmoid
    from develop import Regularizer, L0L1L2
    print ('Using hand hacked model code!\n')
    model = Sequential()
    model.add(Dense(HLN, input_dim=inSize))
    model.add(Activation('sigmoid'))
    #model.add(Gaussian_Noise_For_Sigmoid(std))
    #model.add(Dropout(0.2))
    #model.add(Dense(kernel_initializer='uniform', units=sizeOfOutput))
    model.add(Dense(kernel_initializer ='uniform', units=sizeOfOutput, activity_regularizer=L0L1L2(l1=0.01)))
    model.add(Activation('sigmoid'))
    #model.add(Activation('sigmoid'))
else:
    model = d.ModelBuilder.add_init_layer(inSize, activationFunction=activation_function, layerSize=HLN) \
        .add_output_layer(outputSize=sizeOfOutput, activationFunction='sigmoid')

#    .add_dense_layer(activationFunction='sigmoid', ) \

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy', 'cosine_proximity'])

#callbacks = [StopMaxedAccuracy(monitor='acc', value=1.0, mode='max')]

##########################################################################
# callbacks to monitor the minimsation process
#########################################################################
#
callback_list = []

# activation callback
callback_list.append(d.activation_history())

## this thing csv_logger writes the mtrics logged out to a csv file
callback_list.append(keras.callbacks.CSVLogger('training.log'))

## checkpoints weights during training
callback_list.append(keras.callbacks.ModelCheckpoint(filepath="./tmp/weights.hdf5", verbose=1, save_best_only=True))

## early stoppping monitor
if doEarlyStopper:
    #callback_list.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None))
    callback_list.append(keras.callbacks.EarlyStopping(monitor='acc', patience=10000, verbose=1, mode='auto'))
    #callback_list.append(keras.callbacks.StopMaxedAccuracy(monitor='acc', value=1.0, mode='max'))
##################################################################
# 2. Train the model
##################################################################
#fit = model.fit(X, T, nb_epoch=noOfEpochs, batch_size=32, validation_split=modelValidationDataRatio,
 #               shuffle=True, callbacks=callback_list)
fit = model.fit(X, T, epochs=noOfEpochs, batch_size=32, validation_split=modelValidationDataRatio,
                shuffle=True, callbacks=callback_list)

##################################################################
# 3. Test the model
##################################################################
K.set_learning_phase(0)

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

d.write_out(model, "model.json")


d.write_out(model, "{0}_model.json".format(name))


netStats.save('{0}_stats.h5'.format(name))
model.save('{0}_model.h5'.format(name))



if doPlots:
    import keras.utils
    keras.utils.vis_utils.plot_model(model, to_file='model.png')
    keras.utils.vis_utils.plot_model(model, to_file='{0}_model.png'.format(name))





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
            plt.close()

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
    plt.close()

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



