from __future__ import print_function
import numpy as np
import random
import sys


def genData(size=1000,centroids=10,dimensions=1):
    n = float(size)/centroids
    X = []
    cs = []
    for i in range(centroids):
        c = [random.uniform(-1,1) for i in range(dimensions)]
        cs.append(c)
        s = random.uniform(0.05,0.15)
        x = []
        while len(x) < n:
          a=[np.random.normal(cd,s) for cd in c]
          if len(filter(lambda x: abs(x)>=1, a)) == 0 :
            x.append(a)
        X.extend(x)
    X = np.array(X)[:size]
    print('{0} {1} : {2}'.format(size,centroids,X.shape))
    return X,np.array(cs)

class KMeans(object):
    def __init__(self,X=None,K=None,filename=None,mu=None):
        """
        X the data as an array of arrays
        K the number of centroids to find
        filename - name of file to load from
        mu is the list of centroids
        """
        self._zero()

        # TODO better arg checking
        if X is not None:
            self._init_from_variable(X,K)
        elif filename is not None:
            self._init_from_file(filename)
        else:
            raise ArgumentError('must specify X or filename')

        if mu is not None:
            self._init_mu_from_variable(mu)


        X = self.X
        self.N = len(X)
        try:
            self.dimensions = len(self.X[0])
        except TypeError:
            # one dimensional array of points
            self.dimensions = 1
            self.X = [(x,) for x in X]

    def _zero(self):
        self.mu = None
        self.clusters = None
        self.oldmu = None
        self.oldermu = None
        self.method = None

    def _init_from_variable(self,X,K):
        self.X = X
        self.K = K

    def _init_from_file(self,filename):
        """
        filename the name of a previously saved KMeans object
        """
        # TODO: error handlnig
        with np.load(filename) as handle:
            self._load(handle['data'].tolist())

    def _init_mu_from_variable(self, mu):
        self.mu = mu

    def _load(self,filehandle):
        self.X = filehandle['X']
        self.K = filehandle['K']
        self.mu = filehandle['mu']
        self.clusters = filehandle['clusters']

    def _get_save_dictionary(self):
        self._build_save_dictionary()
        return self._savedic

    def _build_save_dictionary(self):
        self._savedic={}
        self._savedic['X']=self.X
        self._savedic['K']=self.K
        self._savedic['mu']=self.mu
        self._savedic['clusters']=self.clusters

    def save(self,filename):
        self._build_save_dictionary()
        np.savez(filename, data=self._savedic)

    def _cluster_points(self):
        # assign all the points of X to their nearest centroid.
        clusters = {}
        count = 0
        cluster_to_point_mapping={}
        for x in self.X:
            x = np.array(x)
            keys = [(i[0], np.linalg.norm(x-np.array(self.mu[i[0]]))) \
                             for i in enumerate(self.mu)]
            bestmukey = min(keys, key=lambda t:t[1])[0]
            cluster_to_point_mapping.setdefault(bestmukey, []).append(count)
            print('Count {0}: mu {1}'.format(count, bestmukey))
            count = count + 1
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters
        self.cluster_to_point_mapping=cluster_to_point_mapping

    def _compute_new_centroids(self):
        newmu = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newmu.append(np.mean(self.clusters[k], axis=0))
        self.mu = newmu
        return

    def _has_converged(self):
        if self.oldmu is None:
            # catch first run
            return False
        K = len(self.oldmu)
        return(set([tuple(a) for a in self.mu]) == \
               set([tuple(a) for a in self.oldmu])\
               and len(set([tuple(a) for a in self.mu])) == K)

    def init_centers(self):
        # pick a set of random points to use as the initial K values
        #self.mu = random.sample(self.X, self.K)
        self.mu = np.random.choice(self.X, self.K)
        self.mu.sort()

    def find_centers(self):
        self.init_centers()
        while not self._has_converged():
            self.oldmu = self.mu
            self._cluster_points()
            self._compute_new_centroids()

class KMeansPlusPlus(KMeans):
    #def dist_from_centers(self):
        #self.D2 = np.array([min([np.linalg.norm(x-m)**2 for np.array(m) in self.mu]) for np.array(x) in self.X])

    def _radius(self):
        self.dist_from_centers()
        self.r = max(self.D2)**0.5

    def _choose_next_center(self):
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        idx = np.where(self.cumprobs >= r)[0][0]
        return self.X[idx]

    def init_centers(self):
        #self.mu = random.sample(self.X,1)
        self.mu = np.random.choice(self.X, 1)
        # pick a point, any point
        while len(self.mu) < self.K:
            self.dist_from_centers()
            self.mu.append(self._choose_next_center())
        self.mu.sort()

def a(K,dimensions):
    try:
        return a.mem[(K,dimensions)]
    except KeyError:
        result = None
        if K == 2:
            result = 1-3.0/(4.0*dimensions)
        else:
            previous = a(K-1,dimensions)
            result = previous + (1-previous)/6.0
        a.mem[(K,dimensions)] = result
        return result
a.mem={}

class DetK(KMeansPlusPlus):
    def _zero(self):
        super(DetK,self)._zero()
        self.fs = None
        self.fCentroids = []

    def _build_save_dictionary(self):
        super(DetK,self)._build_save_dictionary()
        self._savedic['fs'] = self.fs
        self._savedic['fCentroids'] = self.fCentroids

    def _load(self,filehandle):
        super(DetK,self)._load(filehandle)
        self.fs = filehandle['fs']
        self.fCentroids = filehandle['fCentroids']

    def fK(self, Skm1=0):
        self.find_centers()
        mu, clusters = self.mu, self.clusters
        #Sk = sum([np.linalg.norm(m-c)**2 for m in clusters for c in clusters[m]])
        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(self.K) for c in clusters[i]])

        if self.K == 1 or Skm1 == 0:
            fs = 1.0
        else:
            fs = Sk/(a(self.K,self.dimensions)*Skm1)
        return fs,Sk,mu

    def _bounding_box(self):
        return np.amin(self.X,axis=0),np.amax(self.X,axis=0)

    def gap(self):
        dataMin,dataMax = self._bounding_box()
        self.init_centers()
        self.find_centers()
        mu, clusters = self.mu, self.clusters
        Wk = np.log(sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
                    for i in range(self.K) for c in clusters[i]]))
        # why 10?
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in self.X:
                Xb.append(random.uniform(dataMin,dataMax))
            Xb = np.array(Xb)
            kb = DetK(K=self.K,X=Xb)
            kb.init_centers()
            kb.find_centers()
            ms,cs = kb.mu, kb.clusters
            BWkbs[i] = np.log(sum([np.linalg.norm(ms[j]-c)**2/(2*len(c)) \
                              for j in range(self.K) for c in cs[j]]))
        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2/float(B))*np.sqrt(1+1/B))
        return Wk, Wkb, sk, ms

    def runFK(self, maxK):
        ks = range(1, maxK+1)
        fs = np.zeros(len(ks))
        fCentroidList = []
        Sk = 0
        for k in ks:
            print('k={}'.format(k))
            self.K=k
            self.init_centers()
            fs[k-1], Sk, centroids = self.fK(Skm1=Sk)
            centroids.sort()
            fCentroidList.append(np.array(centroids))
        self.fs = fs
        self.fCentroids = fCentroidList
        # Now assign the best K centroids
        error=0.15
        bestF = np.argmin(fs)
        if fs[bestF] > (1-error):
            bestF = 0
        self.K=bestF+1
        self.mu = fCentroidList[bestF]
        self._cluster_points()


    def runGap(self, maxK):
        ks = range(1, maxK)
        gCentroidList = []
        Wks,Wkbs,sks = np.zeros(len(ks)+1), np.zeros(len(ks)+1), np.zeros(len(ks)+1)
        for k in ks:
            print('k={}'.format(k))
            self.K=k
            self.init_centers()
            Wks[k-1], Wkbs[k-1], sks[k-1], centroids = self.gap()
            gCentroidList.append(np.array(centroids))
        G = []
        for i in range(len(ks)):
            G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
        self.G = np.array(G)
        self.gCentroids = gCentroidList

    def run(self, maxK, which='both'):
        doF = which is 'f' or which is 'both'
        doGap = which is 'gap' or which is 'both'
        if doF:
            self.runFK(maxK)
        if doGap:
            self.runGap(maxK)



def K_estimator(G, flag='doF', error=0.15):
    """Spews out an estimate for K based on Gap stats of FS"""
    i_laststep=0
    if flag == 'doF':
        estimates=[]
        for n in np.array(range(len(G))):
            if n > 0:
                if G[n] < 1.-error:
                    estimates.append(n)
        if estimates == []:
            # no clusters were good, so data is uniform
            # N.B. we append 0 because that will come out as 1 at the end of hte function!
            estimates.append(0)
        sortedList = [x for (y, x) in sorted(zip(G[estimates], estimates))]
        best_guess = sortedList[0]+1
        return G[sortedList], np.array(sortedList)+1, best_guess
    if flag == 'doGap':
        estimates =[]
        for n in range(len(G)):
            if G[n] < 0:
                i = True
            else:
                i = False
            print('n={0}, i = {1}, i_laststep = {2}'.format(n, i, i_laststep))
            if n > 1:
                if i is not i_laststep:
                    # this is an XOR comparison, if the G-stat crosses the axis then k has been found!
                    estimates.append(n-1)
                    print(estimates)
            i_laststep = i
        if estimates == []:
            # no clusters were good, so data is uniform
            estimates.append(1)
        return estimates

def save_layer(layer, name):
    """ Save all the neurons in a layer into a single npz """
    res={}
    for i,l in enumerate(layer):
        res[str(i)] = l._get_save_dictionary()
    np.savez(name, **res)






#with open('data.npy') as f:
#    egg=np.load(f)
#random.seed()

#k = KMeans(K=3,X=egg)
#k.find_centers()
#print k.mu

#kpp = KMeansPlusPlus(K=3,X=egg)
#for x in range(10):
#    kpp.find_centers()
#    print kpp.mu

#dk = DetK(egg)
#dk.run(10)

#print dk.fs
#print dk.G
#print dk.Wks

#k=np.array([1., 2., 3., 4., 5., 6, 7, 8, 9])

#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.plot(k, dk.fs, 'o', label='fs')
#plt.plot(k, dk.G, '+', label='gap')
#plt.ylabel('error stat')
#plt.xlabel('k')
#plt.legend()
#plt.show()


#kpp = p.DetK(10,X=[egg])
#kpp2 = p.DetK(10,N=384)
#kpp2.run(10)
#kpp.run(10)
#kpp.run(10)
#kpp.plot_all()

#kplusplus = KPlusPlus(5, N=200)
#kplusplus.init_centers()
#kplusplus.plot_init_centers()

