import math
from itertools import chain
__author__ = "ella"

## Initilisation I guess
# These are dictionaries containing the values we need
codes = {'A': [0, 1], 'B': [1, 0], 'C': [1, 1], 'D': [0, 0],
         'if then': [1, 1], 'or': [0, 1], 'not both': [1, 0],
         'not': [1], 'pos': [0], 'therefore': [],
         'MP': [0, 1], 'MT': [1, 0], 'AS': [1, 1], 'DS':[0, 0], 'valid': [1], 'invalid': [0]}

classification= dict(MP=[0, 1], MT=[1, 0], AS=[1, 1], DS=[0, 0], valid=[1], invalid=[0], false=[0, 0], \
                     true=[1, 1], unknown=[0, 1])



# Here we
#logicInputs = codes['pos'] + codes['A'] + codes['if then'] + codes['not'] + codes['C'] + codes['not'] + codes['C'] \
            #  + codes['not'] + codes['A']
#logicTargets = classification['MP'] + classification['invalid']

egg=codes['pos']+codes['A']+codes['if then']+codes['not']+codes['C']+codes['not']+codes['C']

print(codes['pos']+codes['A']+codes['if then']+codes['not']+codes['C']+codes['not']+codes['C'])
print(str(egg))
print(str(codes['pos']))


# here we make lists of strings of the possible options for that type of thing
letters = 'ABCD'
sign = ('pos', 'not')
conn = ('if then', 'or', 'not both')
type = ('MP','MT','AS','DS')
output=('valid','invalid')

#print(str(letters))

f = open('workfile.txt', 'w')
g = open('logic_puzzles.txt', 'w')
h = open('logic_answers.txt', 'w')
cl = open('logic_classes.txt', 'w')

# Here we make all the options of inputs, we have 8 options, not 4 as not-A and A are different usw
i = 0;
P,Q = [],[]
for n in sign:
    for p in letters:
        P.append((n, p))
        Q.append((n, p))


# make all the MP problems/
P, Q, =[],[]
for p in letters:
    P.append(((sign[0], p),(sign[1], p)))
Q=P;

# Modus ponens and Modus tollens
for p in range(-1,len(P)-1):
        for q in range(len(Q)):
            if P[p] !=Q[q]:
                #s1 = [P[p], c, Q[q]]
                c = conn[0]
                pM, qM = 0, 0
                for Pp in range(2):
                    pM = pM + 1
                    for Qq in range(2):
                        qM = qM + 1
                        s1 = [P[p][pM % 2], c, Q[q][qM % 2]]
                        s2p = [P[p][pM % 2], P[p][(pM+1) % 2]]
                        concp=[Q[q][qM % 2], Q[q][(qM+1)%2]]
                        truthp= [output[0], output[1]]
                        s2t = [Q[q][qM % 2], Q[q][(qM + 1) % 2]]
                        conct = [P[p][pM % 2], P[p][(pM + 1) % 2]]
                        trutht = [output[1], output[0]]
                        reason=['True','False', 'Unknown','Unknown']
                        namet = ['MP','MT']
                        # this is modus ponens??
                        for a in range(2):
                            temp = [P[p][pM % 2], [c], Q[q][qM % 2], s2p[a], ['therefore'], concp[a], [namet[a],truthp[a]]]
                            new_d = list(chain.from_iterable(temp))
                            print(new_d)
                            for i in range(len(temp)):
                                f.write(str(temp[i]))
                                f.write(' ')
                            egg=[]
                            for item in new_d:
                                egg.append(codes[item])
                            egg = list(chain.from_iterable(egg))
                            for item in egg[0:14]:
                                g.write(str(item)+',')
                            for item in egg[14:17]:
                                h.write(str(item)+',')
                            f.write("\n")
                            g.write("\n")
                            h.write("\n")
                        # this is modus tollens
                        for a in range(2):
                            temp = [P[p][pM % 2], [c], Q[q][qM % 2], s2t[a], ['therefore'], conct[a], [namet[a],trutht[a]]]
                            new_d = list(chain.from_iterable(temp))
                            print(new_d)
                            for i in range(len(temp)):
                                f.write(str(temp[i]))
                                f.write(' ')
                            egg=[]
                            for item in new_d:
                                egg.append(codes[item])
                            egg = list(chain.from_iterable(egg))
                            for item in egg[0:14]:
                                g.write(str(item)+',')
                            for item in egg[14:17]:
                                h.write(str(item)+',')
                            f.write("\n")
                            g.write("\n")
                            h.write("\n")

                    # end Pp loop

                        #f.write("\n")
                    #f.write("\n")

# alternative syllogism (OR)
for p in range(-1, len(P) - 1):
    for q in range(len(Q)):
        if P[p] != Q[q]:
            # s1 = [P[p], c, Q[q]]
            c = conn[1]
            pM, qM = 0, 0
            for Pp in range(2):
                pM = pM + 1
                for Qq in range(2):
                    qM = qM + 1
                    s1 = [P[p][pM % 2], c, Q[q][qM % 2]]
                    s2p = [P[p][(pM + 1) % 2], Q[q][(qM +1) % 2], P[p][pM % 2], Q[q][qM % 2]]
                    concp = [Q[q][qM % 2], P[p][pM % 2], Q[q][(qM + 1) % 2], P[p][(pM + 1) % 2]]
                    truthp = [output[0], output[0], output[1], output[1]]
                    for a in range(4):
                            temp = [P[p][pM % 2], [c], Q[q][qM % 2], s2p[a], ['therefore'], concp[a], ['AS',truthp[a]]]
                            new_d = list(chain.from_iterable(temp))
                            print(new_d)
                            for i in range(len(temp)):
                                f.write(str(temp[i]))
                                f.write(' ')
                            egg=[]
                            for item in new_d:
                                egg.append(codes[item])
                            egg = list(chain.from_iterable(egg))
                            for item in egg[0:14]:
                                g.write(str(item)+',')
                            for item in egg[14:17]:
                                h.write(str(item)+',')
                            f.write("\n")
                            g.write("\n")
                            h.write("\n")

# disjunctive syllogism (NAND)
for p in range(-1, len(P) - 1):
    for q in range(len(Q)):
        if P[p] != Q[q]:
            # s1 = [P[p], c, Q[q]]
            c = conn[2]
            pM, qM = 0, 0
            for Pp in range(2):
                pM = pM + 1
                for Qq in range(2):
                    qM = qM + 1
                    s1 = [P[p][pM % 2], c, Q[q][qM % 2]]
                    s2p = [P[p][(pM + 1) % 2], Q[q][(qM +1) % 2], P[p][pM % 2], Q[q][qM % 2]]
                    concp = [Q[q][qM % 2], P[p][pM % 2], Q[q][(qM + 1) % 2], P[p][(pM + 1) % 2]]
                    truthp = [output[1], output[1], output[0], output[0]]
                    for a in range(4):
                            temp = [P[p][pM % 2], [c], Q[q][qM % 2], s2p[a], ['therefore'], concp[a], ['DS',truthp[a]]]
                            new_d = list(chain.from_iterable(temp))
                            print(new_d)
                            for i in range(len(temp)):
                                f.write(str(temp[i]))
                                f.write(' ')
                            egg=[]
                            for item in new_d:
                                egg.append(codes[item])
                            egg = list(chain.from_iterable(egg))
                            for item in egg[0:14]:
                                g.write(str(item)+',')
                            for item in egg[14:17]:
                                h.write(str(item)+',')
                            f.write("\n")
                            g.write("\n")
                            h.write("\n")

f.close()
g.close()
h.close()

with open('logic_answers.txt') as w:
    for line in w:
        if line=='0,1,1,\n':
            cl.write('1,0,0,0,0,0,0,0,\n')
        elif line=='0,1,0,\n':
            cl.write('0,1,0,0,0,0,0,0,\n')
        elif line=='1,0,1,\n':
            cl.write('0,0,1,0,0,0,0,0,\n')
        elif line=='1,0,0,\n':
            cl.write('0,0,0,1,0,0,0,0,\n')
        elif line=='1,1,1,\n':
            cl.write('0,0,0,0,1,0,0,0,\n')
        elif line=='1,1,0,\n':
            cl.write('0,0,0,0,0,1,0,0,\n')
        elif line=='0,0,1,\n':
            cl.write('0,0,0,0,0,0,1,0,\n')
        elif line=='0,0,0,\n':
            cl.write('0,0,0,0,0,0,0,1,\n')

cl.close()
