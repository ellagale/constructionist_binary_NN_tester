import numpy as np
import random


def make_random_codes(M, n, X=[], weight=[0.5], k=2,symbolList=None, verbose=False):
    """This function makes k-ary random codes of any length"""
    "M is the number of codewords"
    "n is the length of the code"
    "symbolList is the symbols we're using"
    "k is the number of symbols, if symbolList is empty we do standard k-ary logic"
    "X is the training set, so we can use this code to generate nice new test codes"
    if symbolList != None:
        pass
    else:
        symbolList = range(k)
        symbolList = [i for i in range(k)]
        if k ==2:
            binaryFlag=True
        else:
            binaryFlag=False
    noOfSymbols=[]
    if all([divmod(i*n,1)[1]!=0.0 for i in weight]):
        if verbose == True:
            print('Error: weight does not give an integer no of symbols!')
            print('make_random_codes received weight{0}, codelength{1}'.format(weight, n))
    elif sum(weight) != 1.0:
        if verbose == True:
            print('Weight: {0}, does not sum to 1!'.format(weight))
        if k-len(weight) == 1:
            # other symbol weight is defined
            weight.append(1-sum(weight))
            if verbose == True:
                print('We got this, assuming weight = {}'.format(weight))
    for i in range(len(symbolList)):
        noOfSymbols.append(int(n*weight[i]))
#
    #symbolWeights = zip(symbolList, noOfSymbols)
    #print([x for x in symbolWeights])
#
#
    C_base = set([tuple(x) for x in X]) # this makes a set of the points in C - as tuples cos sets need tuples
#
    #line = [symbolList]
    line = [] # this code chunks makes a list of points
    for i in range(len(symbolList)):
        for j in range(noOfSymbols[i]):
            line.append(symbolList[i])
    #line = [symbol for symbol, weight in symbolWeights for i in range(len(weight))]
    random.shuffle(line) # shuffle the line so we dont end up with [0,0,..0,1,1,..,1] in each experiment
    C_all = set([tuple(x) for x in X])
    target_length = M + len(C_base)
#)
    while len(C_all) < target_length:
        print('{}/{}'.format(len(C_all), target_length))
        random.shuffle(line)
        C_all.add(tuple(line))
#
    C = C_all - C_base
    output = np.array([x for x in C])
#
#
#     C = []
#     line = [symbol for symbol, weight in symbolWeights for i in range(weight)]
#     random.shuffle(line)
#     line_length=len(line)
#     print(line)
# #
#     while len(C) < M:
#         print('{}/{}'.format(len(C),M))
#         # loop until we have M lines
#         # makes a list of the correct number of symbols in order
#         # shuffles the line
#         #while (line in C) or (line in X):
#         while line in C or line in X:
#             # if current random order is in C, shuffle again,
#             line = random.sample(line, line_length)
#             #random.shuffle(line)
#             if verbose:
#                 print(line)
#                 print('Is it in C? {}, Is it in X? {}'.format(line in C, line in X))
#             # if we exit the loop here, line is not in C
#         C.append(line)
# #
#     C = np.array(C) # make it a matrix
    #print('oli')
    return output

def make_prototype_codes(M, n, setting=1, weight=[0.5], k=2,symbolList=[1,0], verbose=False):
    """This function makes k-ary random codes from a prototype template"""
    'setting: 1 = perpendicular diagonal vectors'
    "M is the number of codewords"
    "n is the length of the code"
    "symbolList is the symbols we're using"
    "k is the number of symbols, if symbolList is empty we do standard k-ary logic"
    "N.B. reverse the order of symbolList if using binary!"
    # there are of course make prototypes possible!
    if setting == 1:
        # we want perpendicular vectors that span the whole space
        # currently ths only works with binary!
        blockLength = n//M
        noOfBlocks = n//blockLength
        C=[]
        blockTypeList = []
        for b in range(k):
            if symbolList == None:
                blockTypeList.append(np.ones(blockLength)*b)
            else:
                blockTypeList.append(np.ones(blockLength)*symbolList[b])
        j = 0
        for w in range(M):
            t = []
            for i in range(noOfBlocks):
                if j == i:
                    t.append(blockTypeList[0])
                else:
                    t.append(blockTypeList[1])
            j=j+1
            if j == 1:
                if verbose == True:
                    print('Example prototype code: {}'.format(t))
            C.append(np.array(t).reshape(n))
        C = np.array(C)  # make it a matrix
        return C

#To-do: check the weights of random codes, Im not sure the weight param is being used correctly by make_random_codes 22/03

def make_prototyped_random_codes(M, n, p, weight=[0.5], k=2,symbolList=None, verbose=False, decay_templates=0.0):
    """This function makes k-ary random codes from a prototype template"""
    "N.B. hacky !"
    "p is the number of prototypes we have"
    "decay_templates is the prob. that a 1 will be randomly switched to a 0"
    # first get prototypes
    P = make_prototype_codes(M=p, n=n, setting=1, k=2,symbolList=[1,0], verbose=verbose)
    blockLength = n // p
    noOfBlocks = n // blockLength
    newN = n - blockLength
    noOfExamples = M//p
    if verbose:
        print('We have {0} prototypes for {1} codes, giving {2} examples of each'.format(p, M, noOfExamples))
        print('Prototypes have {0} blocks of length {1} each'.format(noOfBlocks, blockLength))
        print('decaying prototype by {}%'.format(decay_templates * 100))
    # we're going to use the prototypes as a mask:
    R = make_random_codes(M, newN, weight=weight, k=2,symbolList=[1,0], verbose=True)
    # we know that all rows of R are non equal, so once they are sliced up and put into C,
    # even with prototype bloc inserted thye will not be equal
    C = np.zeros([M, n])
    for z in range(p):
        Temp=np.zeros([noOfExamples, n])
        #loop over hte number of prototype classes
        for w in range(noOfExamples):
            # loop over word
            j = 0
            for i in range(n):
                # loop over letters
                if P[z,i] != 1:
                    # if P is zero, replace it with part of hte random template
                    # count over j
                    #print('w={0}, i={1}, j={2}, noOfExamples={3}, n={4}'.format(w, i, j, noOfExamples, n))
                    Temp[w,i] = R[w,j]
                    j=j+1
                else:
                    # if P is 1 copy it in NTS could copy in something else here
                    if symbolList == None:
                        if decay_templates!=0:
                            # probability to switch 1 by
                            # todo fix this!
                            # probability to switch a one by
#
                            p0 = random.uniform(0,1)
                            if p0 < decay_templates:
                                # switch a 1 in prototype subset for a 0
                                Temp[w, i] = 0
                            else:
                                Temp[w, i] = P[z, i]
                        else:
                            # no decay, just put in the prototype
                            Temp[w, i] = P[z, i]
                    else:
                        # as above but use whatever symbol was in the symbol list
                        if decay_templates != 0:
                            # probability to switch a one by
                            if verbose == True:
                                print('Decaying prototype by {}%'.format(decay_templates*100))
                            p0 = random.uniform(0,1)
                            if p0 < decay_templates:
                                # switch a 1 in prototype subset for a 0
                                Temp[w, i] = symbolList[1]
                            else:
                                # dont switch
                                Temp[w, i] = symbolList[0]
                        else:
                            # no decay, prototype subfield intact
                            Temp[w, i] = symbolList[0]
        C[0+z*noOfExamples:noOfExamples+z*noOfExamples,:] = Temp
    #print("decayed")
    return C

def make_test_prototyped_random_codes(M, n, p, X=[], weight=[0.5], k=2,symbolList=None, verbose=False, decay_templates=0):
    """Quick hack of make_prototyped_random_codes to generate unseen codes with the same pattern"""
    "N.B. hacky !"
    "p is the number of prototypes we have"
    "decay_templates is the prob. that a 1 will be randomly switched to a 0"
    # first get prototypes
    P = make_prototype_codes(M=p, n=n, setting=1, k=2,symbolList=[1,0], verbose=verbose)
    blockLength = n // p
    noOfBlocks = n // blockLength
    newN = n - blockLength
    noOfExamples = M//p
    if verbose == True:
        print('We have {0} prototypes for {1} codes, giving {2} examples of each'.format(p, M, noOfExamples))
        print('Prototypes have {0} blocks of length {1} each'.format(noOfBlocks, blockLength))
        print('decaying prototype by {}%'.format(decay_templates * 100))
    # we're going to use the prototypes as a mask:
    R = make_random_codes(M, newN, X=X, weight=weight, k=2, symbolList=[1,0], verbose=True)
    # we know that all rows of R are non equal, so once they are sliced up and put into C,
    # even with prototype bloc inserted they will not be equal
    C = np.zeros([M, n])
    for z in range(p):
        Temp=np.zeros([noOfExamples, n])
        #loop over hte number of prototype classes
        for w in range(noOfExamples):
            print(noOfExamples)
            # loop over word
            j = 0
            for i in range(n):
                # loop over letters
                if P[z,i] != 1:
                    # if P is zero, replace it with part of hte random template
                    # count over j
                    print('w={}, i={}, j={}, noOfExamples={}, n={}'.format(w, i, j, noOfExamples, n))
                    Temp[w,i] = R[w,j]
                    j=j+1
                else:
                    # if P is 1 copy it in NTS could copy in something else here
                    if symbolList == None:
                        if decay_templates!=0:
                            # probability to switch 1 by
                            # todo fix this!
                            # probability to switch a one by
#
                            p0 = random.uniform(0,1)
                            if p0 < decay_templates:
                                # switch a 1 in prototype subset for a 0
                                Temp[w, i] = 0
                            else:
                                Temp[w, i] = P[z, i]
                        else:
                            Temp[w, i] = P[z, i]
                    else:
                        if decay_templates != 0:
                            # probability to switch a one by
                            if verbose == True:
                                print('Decaying prototype by {}%'.format(decay_templates*100))
                            p0 = random.uniform(0,1)
                            if p0 < decay_templates:
                                # switch a 1 in prototype subset for a 0
                                Temp[w, i] = symbolList[1]
                            else:
                                # dont switch
                                Temp[w, i] = symbolList[0]
                        else:
                            # no decay, prototype subfield intact
                            Temp[w, i] = symbolList[0]
        C[0+z*noOfExamples:noOfExamples+z*noOfExamples, :] = Temp
    print("decayed")
    return C

def check_duplicate_codewords(C):
    """Check if there are two codewords in C"""
    duplicate_list=[]
    n=0
    for w in range(len(C)):
        for w2 in range(len(C)):
            if w < w2:
                if w is not w2:
                    if (C[w] == C[w2]).all():
                        print('Duplicates detected! C[{0}] = C[{1}]'.format(w, w2))
                        print('C[{0}] = {1}'.format(w, C[w]))
                        print('C[{0}] = {1}'.format(w2, C[w2]))
                        duplicate_list.append([w, w2])
                        n = n+1
    print('{} duplicates found'.format(n))
    return duplicate_list



def hamming_distance(a, b):
    distance = 0
    for i in range(len(a)):
        if a[i]!=b[i]:
            distance += 1
    return distance

def min_hamming_distance(code):
    minHammingDistance = len(code[0])
    for a in code:
        for b in code:
            if (a != b).any():
                tmp = hamming_distance(a, b)
                if tmp < minHammingDistance:
                    minHammingDistance = tmp
    return minHammingDistance

def weight(code, binary=True, verbose=False):
    if binary == True:
        weights=[]
        for word in code:
            weights.append(sum(word))
            min_weight = min(weights)
            max_weight = max(weights)
            ave_weight = np.mean(weights)
            # could add in dispersion and other stuff here
    else:
        pass
    if verbose == True:
        if min_weight == max_weight:
            print('All weights are the same and equal to {}'.format(min_weight))
        else:
            print('min_weight = {0}, max_weight = {1}, ave_weight ={2}'.format(min_weight, max_weight, ave_weight))
    return weights, min_weight, max_weight, ave_weight

def orderedSampleWithoutReplacement(seq, k):
    if not 0<=k<=len(seq):
        raise ValueError('Required that 0 <= sample_size <= population_size')

    numbersPicked = 0
    for i,number in enumerate(seq):
        prob = (k-numbersPicked)/(len(seq)-i)
        if random.random() < prob:
            yield number
            numbersPicked += 1

def combine_train_test_set(X, Test_X):
    """Little function to make a code of both train and test sets"""
    noOfTrainData = len(X)
    noOfTestData = len(Test_X)
    lenOfInput=len(X[3])
    All_X = np.zeros([noOfTrainData+noOfTestData,lenOfInput])
    All_X[range(noOfTrainData),:] = X
    All_X[range(noOfTrainData, noOfTrainData+noOfTestData), :] = Test_X
    return All_X

# def get_test_x(X, noOfTestData, lenOfInput, noOfPrototypes, weight, k=2, symbolList=None, verbose=verbose, decay_templates=0.0, count=10):
#     """Recursive function to find a disjoint test set
#     N.B. the chance of Test_X overlapping with X is very small"""
#     noOfTries=0
#     Test_X = code.make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=weight,
#                                       k=k, symbolList=symbolList, verbose=verbose, decay_templates=decay_templates)
#     if noOfTries > count:
#         print('Error! Unable to find a disjoint Test set!')
#         return Test_X, All_X
#     All_X = combine_train_test_set(X, Test_X)
#     duplicate_list = check_duplicate_codewords(All_X)
#     if not duplicate_list == []:
#         print('Error! Duplicates found')
#         noOfTries = noOfTries + 1
#         get_test_x(X=X, noOfTestData=noOfTestData, lenOfInput=lenOfInput, noOfPrototypes=noOfPrototypes,
#                    weight=weight, k=2, symbolList=None, verbose=verbose, decay_templates=decay_templates)
#     return Test_X, All_X

# def combine_train_test_set(X, Test_X):
#     """Little function to make a code of both train and test sets"""
#     noOfTrainData = len(X)
#     noOfTestData = len(Test_X)
#     All_X = np.zeros([noOfTrainData+noOfTestData,lenOfInput])
#     All_X[range(noOfTrainData),:] = X
#     All_X[range(noOfTrainData, noOfTrainData+noOfTestData), :] = Test_X
#     return All_X

def get_test_x(X, noOfTestData, lenOfInput, noOfPrototypes, weight, k=2, symbolList=None, verbose=False, decay_templates=0.0, count=10):
    """Recursive function to find a disjoint test set
    N.B. the chance of Test_X overlapping with X is very small"""
    noOfTries=0
    Test_X = make_prototyped_random_codes(M=noOfTestData, n=lenOfInput, p=noOfPrototypes, weight=weight,
                                      k=k, symbolList=symbolList, verbose=verbose, decay_templates=decay_templates)
    if noOfTries > count:
        print('Error! Unable to find a disjoint Test set!')
        return Test_X, All_X
    All_X = combine_train_test_set(X, Test_X)
    duplicate_list = check_duplicate_codewords(All_X)
    if not duplicate_list == []:
        print('Error! Duplicates found')
        noOfTries = noOfTries + 1
        get_test_x(X=X, noOfTestData=noOfTestData, lenOfInput=lenOfInput, noOfPrototypes=noOfPrototypes,
                   weight=weight, k=2, symbolList=None, verbose=verbose, decay_templates=decay_templates)
    return Test_X, All_X

def get_test_t(T, noOfPrototypes, noOfTestData=[], lenOfOutput=[], verbose=False):
    """function to make a Test_T matrix to match the Test_X"""
    # things we calculate
    if noOfTestData == []:
        noOfTestData = len(T)
    if lenOfOutput == []:
        lenOfOutput=len(T[0])
    noOfTrainData = len(T)
    noOfExamples = noOfTrainData // noOfPrototypes
    noOfNewExamples = noOfTestData // noOfPrototypes
    # this grabs teh indices where prototype code cahgnes in T
    train_T_indices = [x for x in range(0, noOfTrainData, noOfExamples)]
    # this tells us where the prototype codes should change in T_Test
    test_T_indices = [x for x in range(0, noOfTestData, noOfNewExamples)]
    # we fine the prototype codes from T
    prototypeOutputCodes = T[train_T_indices]
    # make a dictionary of the indices of the change point, for use in list comp. later
    pOC_dict = {}
    for i in range(noOfPrototypes):
        pOC_dict[test_T_indices[i]] = prototypeOutputCodes[i]
    ## now we build Test_T
    Test_T = []
    current_key, new_key = 0, 0
    for i in range(noOfTestData):
        # this line figures out which prototype key to use
        current_key = max([j for j in test_T_indices if j <= i])
        if not current_key == new_key:
            if verbose:
                print('i={}, current_key={}'.format(i, current_key))
                print('Class {} has code: {}'.format(i, pOC_dict[current_key]))
            new_key = current_key
        Test_T.append(pOC_dict[current_key])
    Test_T = np.array(Test_T)
    return Test_T, prototypeOutputCodes

