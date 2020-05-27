from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import h5py

import copy
import inspect
import types as python_types
import warnings
import collections
import develop as d

from keras import backend as K
from keras import activations
#from keras import initializations
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.engine import Layer
#from keras.engine import Merge
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
#from keras.utils.generic_utils import get_from_module

import matplotlib.pyplot as plt
import os

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.engine.topology import Layer
from keras import activations, regularizers, constraints
from keras.optimizers import SGD
import keras.backend as K
from analysis import HelloWorld, jitterer


import six
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object

"""This module is for developing the machinery required to make neural nets and analyse local and global codes

This module does stuff.
"""

__version__ = '0.1'
__author__ = 'Ella Gale'
__date__ = 'Jan 2017'

## To-do
# 0 you wer doing the spotty plots!
# 1. don't forget to set sensible weights if you decide to start getting actual data
# 2. write out file function and a latex output report file


### classes!:

class activation_history(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
            self.activations = []

    def on_batch_end(self, batch, logs={}):
            self.activations.append(logs.get('loss'))

    def on_train_end(self,logs={}):
            self.activations.append(logs.get('loss'))

class Value_Function_Layer(Layer):
    """Just your regular densely-connected NN layer.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # this is equivalent to the above:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.
    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Value_Function_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Value_Function_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Gaussian_Noise_For_Sigmoid(Layer):
    """Apply additive 0.5-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    @interfaces.legacy_gaussiannoise_support
    def __init__(self, stddev, **kwargs):
        super(Gaussian_Noise_For_Sigmoid, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.5,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(Gaussian_Noise_For_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TrainingData(object):
    def __init__(self, dataset, inputSize, outputSize, testDataSize, shuffle=True, validationDataRatio=0.3):
        self.dataset = dataset
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.testDataSize = testDataSize
        self.shuffle = shuffle
        self.validationDataRatio = validationDataRatio

        # check dataset size
        if dataset[[1]].size != inputSize + outputSize:
            raise Exception(" Error! NN input {0} and output dimension do not match the data {1}!!".format(dataset[[1]].size, inputSize + outputSize))

        dataSize = len(dataset)
        validationDataSize = int(dataSize * validationDataRatio)
        trainDataSize = dataSize - (validationDataSize + testDataSize)

        self.dataSize = dataSize
        self.validationDataSize = validationDataSize
        self.trainDataSize = trainDataSize

        if shuffle:
            np.random.shuffle(self.dataset)

        self.XTrain = self.dataset[0:trainDataSize, 0:inputSize]
        self.TTrain = self.dataset[0:trainDataSize, inputSize:inputSize+outputSize]

        if validationDataSize>0:
            self.XValid = self.dataset[0:validationDataSize, 0:inputSize]
            self.TValid = self.dataset[0:validationDataSize, inputSize:inputSize+outputSize]
        else:
            self.XValid, self.TValid = [], []

        if testDataSize>0:
            self.XTest = self.dataset[0:testDataSize, 0:inputSize]
            self.TTest = self.dataset[0:testDataSize, inputSize:inputSize+outputSize]
        else:
            self.XTest, self.TTest = [], []

        self.allInputData = np.append(self.XTrain, self.XTest, axis=0)
        self.allOutputData = np.append(self.TTrain, self.TTest, axis=0)

    def __str__(self):
        result = """We have {0} datapoints:
                    - {1} validation datapoints.
                    - {2} test datapoints.
                    - {3} training datapoints.
                 """.format(self.dataSize, self.validationDataSize, self.testDataSize, self.trainDataSize)

        result = result + "Example Training data:\n{0} --> {1}".format(self.dataset[0][0:self.inputSize],self.dataset[0][self.inputSize:self.inputSize+self.outputSize])
        result = result + "Training shapes (X/T): {0},{1}".format(self.XTrain.shape,self.TTrain.shape)

        temp = sum(sum(self.TTrain)) / len(self.TTrain)

        if temp == 1:
            coding = "Using 1-HOT target encoding."
        else:
            coding = "Target codes are {0} 1s, on average".format(temp)
        result = result + coding
        return result

    def save(self, filename):
        with h5py.File(filename,'w') as fh:
            fh.create_dataset('dataset',data=self.dataset)
            fh.attrs['inputSize'] = self.inputSize
            fh.attrs['outputSize'] = self.outputSize
            fh.attrs['testDataSize'] = self.testDataSize
            fh.attrs['shuffle'] = self.shuffle
            fh.attrs['validationDataRatio'] = self.validationDataRatio

    @classmethod
    def load(cls, filename):
        with h5py.File(filename,'r') as fh:
            dataset = fh['dataset'][:]
            inputSize = fh.attrs['inputSize']
            outputSize = fh.attrs['outputSize']
            testDataSize = fh.attrs['testDataSize']
            shuffle = fh.attrs['shuffle']
            validationDataRatio = fh.attrs['validationDataRatio']
        result = cls(dataset, inputSize, outputSize, testDataSize, False, validationDataRatio)
        result.shuffle = shuffle
        return result


class NetworkStats(object):
    def __init__(self, model, data):
        self.stats = []
        if model is None:
            return

        for layer in model.layers:
            layer_output = K.function([model.layers[0].input],[layer.output])
            self.stats.append(layer_output([data])[0])

    def save(self, filename):
        with h5py.File(filename,'w') as fh:
            for i,layer in enumerate(self.stats):
                fh.create_dataset(str(i),data=layer)

    @classmethod
    def load(cls, filename):
        result = cls(None,None)
        with h5py.File(filename,'r') as fh:
            for i in range(len(fh)):
                result.stats.append(fh[str(i)][:])
        return result

class ModelBuilder():
    def __init__( self, layerSize ):
        self.model=Sequential()
        self.layerSize=layerSize

    @classmethod
    def load(cls, filename):
        """ wrapper around load_model that understands our custom layers """
        return keras.models.load_model(filename,custom_objects={'Value_Function_Layer':Value_Function_Layer, 'Gaussian_Noise_For_Sigmoid': Gaussian_Noise_For_Sigmoid})

    @classmethod
    def add_init_layer( cls, inSize, initialValues='uniform', activationFunction='sigmoid', name='layer_1', layerSize=10 ):
        self=cls(layerSize)
        self.model.add(Dense(layerSize, input_dim=inSize, init=initialValues, activation=activationFunction))
        return self

    def add_dense_layer(self, initialValues='uniform', activationFunction='sigmoid', layerSize=None, ):
        if layerSize==None:
            layerSize=self.layerSize
        self.model.add(Dense(layerSize, init=initialValues, name='layer_{0}'.format(1+len(self.model.layers)),
                             activation=activationFunction))
        return self

    def add_value_layer(self, initialValues='uniform', activationFunction='sigmoid', layerSize=None, ):
        if layerSize==None:
            layerSize=self.layerSize
        self.model.add(Value_Function_Layer(layerSize, init=initialValues, name='layer_{0}'.format(1+len(self.model.layers)),
                             activation=activationFunction))
        return self

    def add_dropout_layer(self, layerSize=None,
                          dropout=0.5):
        if layerSize==None:
            layerSize=self.layerSize
        self.model.add(Dropout(dropout))
        return self

    def add(self, layer):
        self.model.add(layer)
        return self

    def add_output_layer(self, outputSize, initialValues='uniform', activationFunction='softmax'):
        self.model.add(Dense(init=initialValues, output_dim=outputSize, name='output_layer', activation=activationFunction))
        return self.model






class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L0L1L2(Regularizer):
    """Regularizer for L0, L1 and L2 regularization.
    # Arguments
        l0: float; L0 regulariztion factor
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l0=0., l1=0., l2=0.):
        self.l0 = K.cast_to_floatx(l0)
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l0:
            regularization += K.sum(self.l0 * (K.abs(x)/K.abs(x)))
            #mask = K.any(x, axis = -1, keepdims = True)
            #no_nonzero = K.sum(mask)
            #regularization += K.sum(K.any(x, axis = -1, keepdims = True))*self.l0
             #K.any(K.abs(x)) > 0
             #if K.any(K.abs(x) > 0):
            print('x={}, reg={}'.format(x, self.l0 * 1.))
             #   regularization += self.l0 * 1.
        if self.l1:
            print('L1: x={}, reg={}'.format(x, K.sum(self.l1 * K.abs(x))))
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l0': float(self.l0),
                'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.

def l0(l=0.01):
    return L0L1L2(l0=l)

def l1(l=0.01):
    return L0L1L2(l1=l)


def l2(l=0.01):
    return L0L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L0L1L2(l1=l1, l2=l2)


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier: ' +
str(identifier))

###### END CLASSES! ########

#a
def activation_matrix(y_true, y_pred):
    """This writes out the activation of each layer's neurons"""
    out1 = K.mean(y_pred)
    out2 = K.mean(y_true)
    return{
        'Yo1': out1,
        'Yo2': out2
    }
#b
## build model examples!
def build_model(inSize, outSize):
    global noOfLayers
    noOfLayers = 3
    return ModelBuilder.add_init_layer(inSize).add_dense_layer().add_output_layer(outSize)
#build model examples
def three_layer_1_HOT_out(inSize, outSize):
    global noOfLayers
    noOfLayers = 3
    return d.ModelBuilder.add_init_layer(inSize).add_dense_layer().add_output_layer(outSize)

#build model examples
def four_layer_1_HOT_out(inSize, outSize):
    global noOfLayers
    noOfLayers = 3
    return d.ModelBuilder.add_init_layer(inSize).add_dense_layer().add_output_layer(outSize)


#c
def class_counter(T, verbose=1):
    """This counts the number of types of things and prints it nicely if verbose=1"""
    c = collections.Counter([str(x) for x in T])
    output = ['{0}: {1}'.format(x, c[x]) for x in c]
    output.sort()
    if verbose ==1:
        for x in output: print(x)
    return c

#d
def dataset_slicer(dataset, sizeOfInput, sizeOfOutput, sizeOfTestingData, shuffle=True,
                   sizeOfValidationData=0, modelValidationDataRatio=0.3):
    """Chops up data set according to the settings
    and returns a training, validation and test set"""
    'sizeOfValidationData= size left outside of training for final validation'
    'N.B. Validation options need testing '
    global noOfValidData, noOfTrainData, noOfTestData
    # check data size is sensible
    if dataset[[1]].size != sizeOfInput + sizeOfOutput:
        raise Exception(" Error! NN input {0} and output dimension do not match the data {1}!!"
                        .format(dataset[[1]].size, sizeOfInput + sizeOfOutput))

    # find shit out
    noOfTotalData = len(dataset)
    print('We have {0} datapoints in total :)'.format(noOfTotalData))
    if shuffle == True:
        # Shuffle da data this line shuffles the ROWs and leaves the columns alone
        np.random.shuffle(dataset)

    if sizeOfValidationData < 1:
        # we have a ratio people
        noOfValidData = np.int(sizeOfValidationData * noOfTotalData)
    else:
        # we have a number
        noOfValidData = sizeOfValidationData
    print('We have {0} validation datapoints'.format(noOfValidData))

    if sizeOfTestingData < 1:
        # we have a ratio people
        noOfTestData = np.int(sizeOfTestingData * noOfTotalData)
    else:
        # we have a number
        noOfTestData = sizeOfTestingData
    print('We have {0} test datapoints'.format(noOfTestData))

    noOfTrainData = noOfTotalData - noOfValidData - noOfTestData
    print('Leaving {0} training datapoints'.format(noOfTrainData))

    if sizeOfValidationData != 0:
        print('Warning: Validation data entered manually, is this what you want?')
        temp = sizeOfValidationData - modelValidationDataRatio * noOfTotalData
        if temp != 0:
            print('Warning: {0} datapoints are not being used'.format(temp))

    # now chop up dataset
    XTrain = dataset[0:noOfTrainData, 0:sizeOfInput]
    TTrain = dataset[0:noOfTrainData, sizeOfInput:sizeOfInput + sizeOfOutput]
    if noOfValidData != 0:
        # this will be relevant if/when keras adds validation data!
        XValid = dataset[0:noOfValidData, 0:sizeOfInput]
        TValid = dataset[0:noOfValidData, sizeOfInput:sizeOfInput + sizeOfOutput]
    if noOfTestData !=0:
        XTest = dataset[0:noOfTestData, 0:sizeOfInput]
        TTest = dataset[0:noOfTestData, sizeOfInput:sizeOfInput + sizeOfOutput]
    else:
        XTest, TTest = [],[]
    print('Example training data:\n{0} --> {1}'
              .format(dataset[0][0:sizeOfInput], dataset[0][sizeOfInput:sizeOfInput+sizeOfOutput]))
    print('XTrain and TTrain shapes:{0} --> {1}'
          .format(XTrain.shape, TTrain.shape))

    temp = sum(sum(TTrain)) / len(TTrain)
    if temp == 1:
        print('You are using 1-HOT target encoding')
    else:
        print('Target codes are {0} ones, on average'.format(temp))
    temp = sum(sum(XTrain)) / len(XTrain)
    if temp == 1:
        print('You are using 1-HOT target encoding')
    else:
            print('Input codes are {0} ones, on average'.format(temp))
    return XTrain, TTrain, XTest, TTest


def deprocess_net_image(image):
    # Helper function for deprocessing preprocessed images, e.g., for display.
    """deprocessor image for caffe"""
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

#g

def get_data_path(dirname,filename):
    "This finds the data in ../../data/dirname/filename"
    return os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                        'data',
                        dirname,
                        filename)

def generate_random_data(inSize, outSize, noOfExamples, base):
    """Produces random data as a dataset"""
    "!! Check this works !! "
    data = np.random.randint(base,size=(noOfExamples, inSize))
    labels = np.random.randint(base, size=(noOfExamples, outSize))
    return data, labels

#def build_model(inSize,outSize):
 #  model = Sequential()
  # model.add(Dense(10, input_dim=inSize, init='zero', name='layer_1', activation='sigmoid'))
   #model.add(Dropout(0.5))
   # second layer, 10 fully connected units
   #model.add(Dense(10, init='one', name='layer_2', activation='sigmoid'))
   #model.add(Activation('sigmoid'))
   #model.add(Dropout(0.5))
  # model.add(Dense(init='uniform', output_dim=outSize, name='output_layer', activation='softmax'))
   #model.add(Dense(init='uniform', output_dim=outSize, name='output_layer', activation='sigmoid'))
   #model.add(Activation("softmax"))
#   global noOfLayers
#   noOfLayers=3
#   return model
#l
def load_in(name):
    " load in name and create loaded_model"
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model
    # evaluate loaded model on test data
#m
def make_1HOT(n):
    """Make a list of 1HOT codes of abitrary size"""
    T=[]
    for i in range(n):
        t= np.zeros(n)
        t[i-1] = 1
        T.append(t)
    return T


#p
def prediction_tester(model, X, T, name='', batch_size=32, example_no=8, verbose=1, error=0.1):
    """Gets model predictions and does some basic counting with them"""
    predictions = model.predict(X, batch_size=batch_size)
    predictions_of_classes = model.predict_classes(X, batch_size=batch_size)
    noCorr = 0
    noWithinE = 0
    n = len(X)
    mistakesCorr=[]
    mistakesWithinE=[]

    if error > 0.5:
        print('Error! prediction_tester expects an error value as a decimal in units of the Targets max value ')
        print('prediction_tester received {} as an input error'.format(error))
    print('\n{} stats and example data:'.format(name))
    for i in range(n):
        if len(T[i, :]) < 6:
            if verbose == 1 and i < example_no:
                print('{0}: {1}'.format(T[i, :], predictions[i, :]))
                print('{}'.format(np.round(predictions[i, :])))
        else:
            if verbose == 1 and i <= example_no:
                print('{0}'.format(T[i, :]))
                print('{}'.format(predictions[i, :]))
                print('{}'.format(np.round(predictions[i, :])))
                print('{}'.format(predictions_of_classes[i]))
        if (np.round(predictions[i, :]) == T[i, :]).all():
            # are the predictions within 0.5 integer?
            if verbose == 1 and i <= example_no:
                print(':)\n')
            # if yes, count it
            noCorr = noCorr + 1
        else:
            # if no --> is a mistake!
            mistakesCorr.append(i)
        if (abs(predictions[i, :] - T[i, :]) < error*np.amax(T)).all():
            # are the predictions within error %?, if yes count
            noWithinE = noWithinE + 1
        else:
            # if not, count the mistakes
            mistakesWithinE.append(i)
    pc = 100*float(noCorr)/n
    pcWE = 100 * float(noWithinE) / n
    pcE = 100*error*np.amax(T)
    print('\n{0}: {1} correct, {2} %'.format(name, noCorr, pc))
    print('{0}: {1} correct, {2} %, to within {3} %'.format(name, noWithinE, pcWE, pcE))
    c = class_counter(T, verbose=verbose)
    print ('{} sample classes'.format(len(c)))
    if verbose == 1 and mistakesWithinE != 0:
        print('Mistakes in {}'.format(name))
    elif verbose == 1 and mistakesWithinE == 0:
        print('No mistakes in {}'.format(name))
    class_counter(T[mistakesCorr], verbose=verbose)
    if verbose ==1:
        lost = list(set(mistakesWithinE ) - set(mistakesCorr))
        if len(lost) < 15 and len(lost) != 0:
            print('Example mistakes within 50% but not within error value')
            for x in range(len(lost)):
                print('{0}: {1}'.format(T[lost[x]], predictions[lost[x]]))
        if len(lost) > 15:
            print('Example mistakes within 50% but not within error value')
            for x in range(3):
                print('{0}: {1}'.format(T[lost[x]], predictions[lost[x]]))

#v
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # taken from caffee example
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data);
    plt.axis('off')


#w
def write_out(model, name):
    " serialize model to JSON, output weights as hdf5"
    model_json = model.to_json()
    with open(name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
























#newModel = load_in("model.json")
# evaluate loaded model on test data

