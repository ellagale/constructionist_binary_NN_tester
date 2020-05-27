#!/usr/bin/python
import os
import numbers
import getopt
import sys

# set defaults
noName = 1
includeTop = 0 # don;t include the directiory you call walker ing  (avoids duplicates)
outSize = 0 # call this frm the command line to override and do only a set number of selectivity tests
verbose = 0

# first parse any command line parameters
try:
    opts, args = getopt.getopt(sys.argv[1:],"n:i:o:v:",["name=","includeTop=", "outSize=", "verbose="])
except getopt.GetoptError:
    print("{0}: [-n|--name=<name>] [-i|--includeTop=<include top level dir?>] [-o|--outSize=<output layer size>] [-v|--verbose=<be verbose?>]".format(sys.argv[0]))
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

dirlist = [x[0] for x in os.walk(os.getcwd())]
print(dirlist)
if includeTop != 1:
    dirlist = dirlist[1:]

results = {}

outputFilename = 'output.log'
analysisFilename = 'analysis.log'
select = 'Selecitivy' # code to count on
stop = 'currently on neuron:' # code to stop on if we're counting neurons as well


def isNumber(n):
    try:
        p = float(n)
    except ValueError:
        return False
    return True


def parseDir(directory):
    params = None
    if verbose == 1:
        print(directory)
    with open(os.path.join(directory, outputFilename), 'r') as oFile:
        for line in oFile:
            if line.startswith('Running with'):
                params = line.replace(',', '').split(None)
                if verbose == 1:
                    print(line)
                break
            if line.startswith ('Layer 0 has'):
                # hack to deal with older file format
                #print('old style file')
                if verbose == 1:
                    print(line)
                params = line.replace(' ', ' ').split(None)
                params = ['m','e','h',params[3], params[1], params[6]]
                #print(params)
                break

        if outSize == 0:
            if verbose == 1:
                print(params)
            HLN = int(params[3])
        else:
            HLN = outSize
            if verbose == 1:
                print(HLN)
    if params == None:
        return
    paramLine = 'p'.join([n.replace('.', 'p') for n in params if isNumber(n)])
    count = 0
    neuronCount = 0
    with open(os.path.join(directory, analysisFilename), 'r') as aFile:
        for line in aFile:
            if line.startswith(stop):
                neuronCount = neuronCount + 1
                if neuronCount == HLN+1:
                    #print ('{0} neurons found!'.format(HLN))
                    #print (line)
                    break
            if line.startswith(select):
                count = count + 1
    try:
        results[paramLine].append(count)
    except KeyError:
        results[paramLine] = [count]

    return paramLine, count


for directory in dirlist:
    parseDir(directory)
    # print('{0}:{1}'.format(directory,parseDir(directory)))

for key in results:
    if noName == 0:
        print('{2}p{0} = {{{1} }};'.format(key, ','.join(map(str, results[key])),name))
    else:
        print('p{0} = {{{1} }};'.format(key, ','.join(map(str, results[key]))))


