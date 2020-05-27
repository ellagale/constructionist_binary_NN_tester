#!/usr/bin/python
import getopt
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kmeans
import cluster_stats as cs
import analysis as a
import develop as d

## Analysis script!
## This does not run kmeans or a neural network, instead it expects that you've done that!

# Initial values
HLN = 500
noName = 1
name='Random'
outSize=50
tf_data = False
doLabels = False
h5_filename = None

# first parse any command line parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:H:o:f:T",["name=","HLN=", "outSize=","h5_file=", "isTensorflowData"])
except getopt.GetoptError:
    print("{0}: [-n|--name=<name>] [-H|--HLN=<neuron count>] [-o|--outSize=<output layer size>] [-f|--h5_file=<hdf5 file>]  [-T|--isTensorflowData=False]".format(sys.argv[0]))
    sys.exit(1)
for opt,arg in opts:
    if opt in ('-n', '--name'):
        noName = 0
        name = arg
        print("{}".format(name))
    elif opt in ('-H', '--HLN'):
        HLN = int(arg)
    elif opt in ('-o', '--outSize'):
        outSize = int(arg)
    elif opt in ('-T', '--tensorflow'):
        tf_data = True
        doLabels = True
    elif opt in ('-f', '--h5_file'):
        h5_filename = arg

# TODO: check for invalid arg combos
## hacky fixes revmoe this
h5_filename=None
noOfNeurons = HLN
noOfLayers = 1

print('Running with HLN: {}'.format(HLN))
import pdb
#pdb.set_trace()

################################################################################
## read shit in                                                               ##
################################################################################

# TODO: sensibly read this data in from something created by develop or at least get the data names
# this gets our data
#noName=0
activation_table = None
if h5_filename is not None:
    activation_table = kmeans.ActivationTable(h5_filename)
    noOfLayers = 1
    noOfNeurons = activation_table.neuron_count
    # What does this map to?
    HLN = 2048
else:
    print("This version requries a h5 file")
    #sys.exit(1)

print ('loading DetKs')
# todo; on the fly?
if h5_filename is not None:
    print('Using h5 file')
    AllDks =[[cs.DetK(X=np.array([[x] for x in activation_table.activations[:,i]])) for i in range(noOfNeurons)]]
else:
    print('using npz')
    AllDks = [[cs.DetK(filename='detk_layer0_neuron{0}.npz'.format(i)) for i in range(x,min(HLN,x+100))] for x in range(0,HLN,100)]

dks = [AllDks[0][0:8]]#[AllDks[0][testSet[0]], AllDks[1][testSet[1]], AllDks[2][testSet[2]]]
# Todo Fix this later! If there is a k it can read it, but how to try catch if not?
#if not len(AllDks[0][0].fs)=='None':
#    noOfK =len(AllDks[0][0].fs)
#else:
 #   noOfK = 0
noOfK = 0
doPlots=1 # turns off all plots!
showPlots = 0 # shows plots on screen for manual saving -- use for debugging
savePlots = 1 # saves out the plots n.b. you can both save and show plots
#noOfLayers = len(dks)
doKAnalysis = 0 # turns on or off the Fk plotter
verbose = True

plotFlags = 'SpSs'
# plotFlag options: 'Fk' --> f(k) plotter for all levels
#                   'Sp' --> spotty plotter
#                   'Sg' --> selectivity

## !! NTS this is wrong! Fix it!!
if doKAnalysis == 1:
    for l in range(len(testSet)):
        for j in range(len(testSet[l])):
            print j
            print('dk.fs = {}'.format(dks[l][j].fs))
            FkOut, KOut, best_guess \
                = cs.K_estimator(dks[l][j].fs, flag='doF', error=0.15)
            print('Neuron {0}{1}:'.format(l, j))
            print('F(k) vals: {0}'.format(FkOut))
            print('K tried: {0} --> best k is:{1}\n'.format(KOut, best_guess))

print('y = {}'.format(dks[0][0].fs))

import itertools
##!! NTS this plotting code needs finishing!
## This is the fs plotter!


if doPlots !=0:
    if 'Fk' in plotFlags:
        # This plots of F(k) function, which tells you how groups (k) the code thinks you have!
        # (taken from reference [])
        a.fk_plotter(dks=dks, noOfK=noOfK, showPlots=0, savePlots=1, lRange=[0, 1])
        #zB.: fk_plotter(dks, noOfK, lRange=None, error=0.15, xaxis=1, title=None, xlabel=None, ylabel=None)


      # do comparison here
#egg= [ idx for idx,x in enumerate(allOutputData) if x==current_key]
#[(x[0],random.random()) for x in s.dks[0][0].X]

#[x[0] for x in s.dks[0][0].clusters[2]]

input_flag = 'K'
doHistogram = False

def spotty_plotter(dks, input_flag='K', doHistogram=False, colour='random', doMu=True,showPlots=0, savePlots=1, cols=2, label=''):
    """Make things that look like neuronal plots"""
    "doMu is whether to plot the centers of K-means centroids (mu)"
    "colour= 'random' or 'centroid', or 'black' or ''"
    for l in range(noOfLayers):
        fig=plt.figure(l)
        #t = a.jitterer(out, l)
        # yrange=max(out[l])-min(out[l])
        r=len(dks[l])/cols
        c=cols
        n=1
        if input_flag=='K':
            # new style using dks
            if doHistogram == True:
                pass
            else:
                for i in range(r):
                    for j in range(c):
                        plt.subplot(r, c, n)
                        if colour == 'centroid':
                            #z = np.random.rand(576)
                            cf = dks[l][n - 1].clusters
                            colourList = itertools.cycle(['blue', 'firebrick', 'gray', 'darkgreen', 'm',
                                                          'darkorange', 'black', 'red', 'gold', 'darkcyan',
                                                          'olivedrab', 'dodgerblue'])
                            for cn in range(len(cf)):
                                x_data = [x[0] for x in cf[cn]]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in cf[cn]]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), marker="o", alpha=0.5,
                                            color=colourList.next())
                        else:
                            if colour == 'random':
                                z = np.random.rand(576)
                                x_data = [x[0] for x in dks[l][n - 1].X]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in dks[l][n - 1].X]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), c=z, marker="o", alpha=0.5)
                            if colour == 'black':
                                x_data = [x[0] for x in dks[l][n - 1].X]
                                y_data = [1 + np.random.uniform(-0.25, 0.25) for x in dks[l][n - 1].X]
                                plt.scatter(x_data, y_data, label='neuron ' + str(j), color='k', marker="o", alpha=0.5)
                        if doMu == 1:
                            mu_data = [x[0] for x in dks[l][n - 1].mu]
                            plt.scatter(mu_data, np.ones(len(mu_data)),
                                    marker="^", facecolors='none', edgecolors='k', s=100, alpha=1.0)
                        n=n+1
                        plt.xlim([-0.05, 1.05])
                        plt.ylim([0.7, 1.3])
                        cur_axes = plt.gca()
                        cur_axes.axes.get_yaxis().set_visible(False)
                        cur_axes.axes.get_xaxis().set_visible(False)
        # if input_flag=='noK':
        #     # old style, use out for the activations
        #     # this may be broken!
        #     t = a.jitterer(out, l)
        #     r = t.shape[0] / 2
        #     for i in range(r):
        #         for j in range(c):
        #             plt.subplot(r, c, n)
        #             plt.scatter(out[l][n-1], t[l][n-1], label='neuron ' + str(j), s=80, c=z, marker="o", alpha=0.5)
        #             n=n+1

        if showPlots == 1:
            plt.show()
            plt.close()
        if savePlots == 1:
            fig.savefig('spotty' + str(l) + label + '.png', dpi=fig.dpi)
            plt.close()

def selectivity_grid(dks, input_flag='K', doHistogram=False, colour='random', doMu=True, showPlots=0, savePlots=1):
    """Makes selectivity measures and a grid"""
    "doMu is whether to plot the centers of K-means centroids (mu)"
    "colour= 'random' or 'centroid', or 'black' or ''"
    for l in range(noOfLayers):
        print l
        fig=plt.figure(l)
        #t = a.jitterer(out, l)
        # yrange=max(out[l])-min(out[l])
        r=len(dks[l])/2
        c=2
        n=1
        if input_flag=='K':
            # new style using dks
            if doHistogram == True:
                pass
            else:
                for i in range(r):
                    for j in range(c):
                        plt.subplot(r, c, n)
                        cf = dks[l][n - 1].clusters
                        noOfClusters=len(cf)
                        for cn in range(noOfClusters-1):
                            # NTS this is hacky and only works for a list of 2 things
                            if dks[l][n-1].mu[cn] > dks[l][n-1].mu[cn+1]:
                                old_selectivity = min(cf[cn]) -max(cf[cn + 1])
                            else:
                                old_selectivity = min(cf[cn+1]) -max(cf[cn])
                        if noOfClusters==2:
                            #old school selectivity IS DEFINED, so print it
                            print('layer {0}, neuron {1}, selectivity = {2}'.format(l, n, old_selectivity))
                            z = float(old_selectivity)
                            plt.text(0.05, 0.95, 'sel =' + str(old_selectivity), fontsize=14,
                                     verticalalignment='top')
                            if z < 0.5:
                                plt.text(0.05, 0.95, 'sel =' + str(old_selectivity), fontsize=14,
                                         verticalalignment='top', color='white')
                            if z < 0:
                                z = 0
                            plt.text(0.01, 1.97, str(l)+str(n), fontsize=12,
                                     verticalalignment='top')
                        else:
                            z = 0
                        if z < 0.5:
                            plt.text(0.2, 1.2, 'k = ' + str(dks[l][n - 1].K), fontsize=14,
                                 verticalalignment='top', color='white')
                        else:
                            plt.text(0.2, 1.2, 'k = ' + str(dks[l][n - 1].K), fontsize=14,
                                     verticalalignment='top')
                        n=n+1
                        plt.xlim([-0.05, 1.05])
                        plt.ylim([0.7, 1.3])
                        cur_axes = plt.gca()
                        cur_axes.axes.get_yaxis().set_visible(False)
                        cur_axes.set_axis_bgcolor((z, z, z))


        if showPlots == 1:
            plt.show()
            plt.close()
        if savePlots == 1:
            fig.savefig('spotty' + str(l) + '.png', dpi=fig.dpi)
            plt.close()





#AllDks = AllDks[6]
foundSelectivityList=[]
foundClassList=[]
foundNeuronList=[]


#@profile
def compute_selectivity(at):
    # Usefult things from AT
    # .activation_labels dict, key=labelname, value=list of indices
    # .labels = list of labels
    # .label_mappings[i] = label of point with index i

    # old code:
    # match_sets[label-1-hot] = list of indices with that label
    # notmatch_sets[label-1-hot] = list of indices without that label

    # TODO: remove all dks and generate on-the-fly
    matches = at.activation_labels
    notmatches = {label:[] for label in matches}

    pdb.set_trace()
    try:
        filters={}
        inv_filters={}
        for label in at.labels:
            filters[label] = np.zeros(at.image_count, dtype=bool)
            inv_filters[label] = np.ones(at.image_count, dtype=bool)
            for i in matches[label]:
                filters[label][i] = True
                inv_filters[label][i] = False
    except:
        pdb.set_trace()

            
    label_count = len(filters)
            

    #pdb.set_trace()
    # save having to reallocate every loop
    label_mins = np.zeros(label_count)
    label_maxs = np.zeros(label_count)
    for layer_index,layer in enumerate(AllDks):
        for neuron_index, neuron_detk in enumerate(layer):
            print('currently on layer {}, neuron {}'.format(layer_index,neuron_index))
            act = neuron_detk.X
            overall_min = np.min(act)
            overall_max = np,max(act)
            pdb.set_trace()
            for idx,label in enumerate(filters):
                filtered = act[filters[label]]
                label_mins[idx] = np.min(filtered)
                label_maxs[idx] = np.max(filtered)
            pdb.set_trace()

            for idx,label in enumerate(filters):
                minMatch = label_mins[idx]
                maxMatch = label_maxs[idx]
                if minMatch != overall_min:
                    minNotMatch = overall_min
                else:
                    minNotMatch = np.min(label_mins[inv_filters[label]])
                if maxMatch != overall_max:
                    maxNotMatch = overall_max
                else:
                    maxNotMatch = np.max(label_maxs[inv_filters[label]])
                        

                print('{0} matches: from {1} to {2}'.format(len(matches), minMatch, maxMatch))
                print('{0} matches: from {1} to {2}'.format(len(notmatches), minNotMatch, maxNotMatch))
                if maxMatch < minNotMatch:
                    left=actMatches
                    right=actNotmatches
                elif maxNotMatch < minMatch:
                    left = actNotmatches
                    right = actMatches
                #if min(right) > max(left):

                selectivity = min(right) - max(left)
                print('Selectivity of {0}'.format(selectivity))
                if abs(selectivity) > 0.01:
                    print('Selecitivy of {} found!'.format(selectivity))
                    #print('current_key is: {}'.format(current_key))
                    if doLabels:
                        print('current_label is {}'.format(current_label))
                    foundSelectivityList.append(selectivity)
                    foundClassList.append(current_key)
                    foundNeuronList.append([l,n])
                    x_data=[[act[matches]],[act[notmatches]]]
                    colourList = itertools.cycle(['blue', 'firebrick', 'gray', 'darkgreen', 'm',
                                                                          'darkorange', 'black', 'red', 'gold', 'darkcyan',
                                                                          'olivedrab', 'dodgerblue'])
                    fig=plt.figure()
                    y_data = [1 + np.random.uniform(-0.25, 0.25) for x in matches]
                    plt.scatter(act[matches], y_data, label='Yo!', marker="o", alpha=0.5, c='firebrick')
                    y_data = [1 + np.random.uniform(-0.25, 0.25) for x in notmatches]
                    plt.scatter(act[notmatches], y_data, label='Yo!', marker="*", alpha=0.5)
                    if showPlots == 1:
                        plt.show()
                        plt.close()
                    if savePlots == 1:
                        fig.savefig('digit' + str(l) + str(n) + '.png', dpi=fig.dpi)
                        plt.close()
    print('Neuron, Selectivity, Class')
    for i in range(len(foundSelectivityList)):
        print('{0},{1},{2}'.format(foundNeuronList[i], foundSelectivityList[i], foundClassList[i]))

#pdb.set_trace()
if 'Ss' in plotFlags:
    #compute_selectivity(activation_table)
    pass

if doPlots == 1:
    if 'Sp' in plotFlags:
        if noOfNeurons > 100:
            spotty_plotter(input_flag='K', dks=AllDks, colour='black', savePlots =1, showPlots = 0, cols=5, doMu =0)
        else:
            spotty_plotter(input_flag='K', dks=AllDks, colour='black', savePlots=1, showPlots=0, cols=20, doMu=0)
    if 'Sg' in plotFlags:
        selectivity_grid(AllDks, input_flag='K', doHistogram=False, colour='random', doMu=False, \
                         savePlots =savePlots, showPlots = showPlots)

def histo_scatter(x, y):

    # the random data
    #x = np.random.randn(1000)
    #y = np.random.randn(1000)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    #axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    #axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.0025
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    #axScatter.set_xlim((-lim, lim))
    axScatter.set_xlim((0.7, 1))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    #axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    #axHisty.set_ylim(axScatter.get_ylim())

    plt.show()
    plt.close()

if 'Hs' in plotFlags:
    histo_scatter(out[0][:,1-1].T, t[1-1])
