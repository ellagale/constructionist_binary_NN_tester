#!/usr/bin/python
import os
import getopt
import sys
import bz2
import subprocess

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

results = {} 
hamresults = {}

uncompressedFilename='output.log'
outputFilename='output.log.bz2'
analysisFilename='analysis.log'
select='Selecitivy'

def isNumber(n):
    try:
        p = float(n)
    except ValueError:
        return False
    return True



def parseDir(directory):
    if verbose:
        print(directory)
    params=None
    ham=None
    correct=None
    bzFile = os.path.join(directory,outputFilename)
    unFile = os.path.join(directory,uncompressedFilename)
    anFile = os.path.join(directory,analysisFilename)
    cwd=os.getcwd()
    if os.path.isfile(bzFile) and doCompressedLog:
        with bz2.BZ2File(bzFile, 'r') as oFile:
            corr = []
            # n.b. bz files give you bytes not strings when read
            for byteline in oFile:
                line = str(byteline,'utf-8')
                if line.startswith('Running with'):
                    params=line.replace(',','').split(None)
                    print(params)
                    HLN=params[3]
#                    break
                if line.startswith('Input min Hamming'):
                    ham=line.replace(':' , '').split(None)[-1]
                    print(line)
                    break
    if params==None and os.path.isfile(unFile) and doLog:
        with open(os.path.join(directory,uncompressedFilename),'r') as oFile:
            for line in oFile:
                if line.startswith('Running with'):
                    params=line.replace(',','').split(None)
                    if verbose:
                        print(line)
#                     #break
                if line.startswith('Input min Hamming'):
                    ham=line.replace(':' , '').split(None)[-1]
                    if verbose:
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
        except KeyError:
            results[paramLine] = [ count ]
            hamresults[paramLine] = [ ham ]

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
    else:
        print('p{0} = {{{1} }};'.format(key, ','.join(map(str, results[key]))))
        print('Hamp{0} = {{{1} }};'.format(key, ','.join(map(str, hamresults[key]))))



