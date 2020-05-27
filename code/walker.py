#!/usr/bin/python
import os
import numbers

dirlist=[x[0] for x in os.walk(os.getcwd())]

results = {} 

outputFilename='output.log'
analysisFilename='analysis.log'
select='Selecitivy'

def isNumber(n):
    try:
        p = float(n)
    except ValueError:
        return False
    return True

def parseDir(directory):
    params=None
    with open(os.path.join(directory,outputFilename),'r') as oFile:
        for line in oFile:
            if line.startswith('Running with'):
                params=line.replace(',','').split(None)
                break
    if params==None:
        return
    paramLine='p'.join([n.replace('.','p') for n in params if isNumber(n)])
    count=0
    with open(os.path.join(directory,analysisFilename),'r') as aFile:
        for line in aFile:
            if line.startswith(select):
                count = count +1
    try:
        results[paramLine].append(count)
    except KeyError:
        results[paramLine] = [ count ]

    return paramLine,count
    
for directory in dirlist:
    parseDir(directory)
    #print('{0}:{1}'.format(directory,parseDir(directory)))

for key in results:
    print('p{0} = {{{1} }};'.format(key,','.join(map(str,results[key]))))
