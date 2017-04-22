import sys
import numpy as np

X = np.loadtxt(sys.argv[1], delimiter=',')
Y = np.loadtxt(sys.argv[2], delimiter=',')
X0 = np.loadtxt(sys.argv[3], delimiter=',')

# prior probability of the classes
def ClassPrior(Y):
    
    N = Y.shape[0]
    
    sortedY = np.sort(Y)

    Classes = [sortedY[i] for i in range(N) if sortedY[i]!=sortedY[i-1]]

    ProbofClass = [np.sum(Y == Class)/float(N) for Class in Classes]
        
    return np.array(ProbofClass)

# The class conditional density is a gaussian so let us estimate that. Note that the features are independent from each other, at least we assume that in the naive bayes
def ClassCondDensityParam(X,Y):
    
    N = Y.shape[0]
    sortedY = np.sort(Y)
    Classes = [sortedY[i] for i in range(N) if sortedY[i]!=sortedY[i-1]]
    NClasses = len(Classes)
    NFeatures = X.shape[1]
    
    Mean = np.zeros((NClasses, NFeatures))
    Covar = np.zeros((NClasses, NFeatures, NFeatures))
    
    for Class in Classes:
        
        idx = (Y == Class)
        x = X[idx,:]

        Mean[Class,:] = np.mean(x, axis = 0)
        
        DevFromMean = x-Mean[Class,:]
        Rank1updateDev = np.dot(DevFromMean.T,DevFromMean)
        Covar[Class,:,:] = np.divide(Rank1updateDev,float(x.shape[0]))

    return Mean, Covar


def ClassCondDensityNormal(X, mean, Covar):
    
    DetCov = np.linalg.det(Covar)
    invCov = np.linalg.inv(Covar)
    
    DevFromMean = (X-mean)
    ExpTerm = np.exp(-(0.5) * np.dot(np.dot(DevFromMean,invCov),DevFromMean))
    density = (1.0/np.sqrt(DetCov))*ExpTerm
    
    return density


# Plug in classifier: computes the probability of belonging to a class
def predOfClass(prior, X, Mu, Cov):
    
    N = X.shape[0]
    F = X.shape[1]
    C = Mu.shape[0]
    
    PredictedClass = np.zeros((N,C))

    for points in range(N):
        
        PredictedAllClass = np.zeros(C)
        
        for Class in range(C):
            
            PredictedAllClass[Class] = (prior[Class] * ClassCondDensityNormal(X[points,:], Mu[Class,:], Cov[Class,:,:]))

        PredictedClass[points,:] =   PredictedAllClass/np.sum(PredictedAllClass)
    
    return PredictedClass


Prior =  ClassPrior(Y)
m,c = ClassCondDensityParam(X,Y)
pred = predOfClass(Prior, X0, m, c)

np.savetxt('probs_test.csv', pred, delimiter=',')