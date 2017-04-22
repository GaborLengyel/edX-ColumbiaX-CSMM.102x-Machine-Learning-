import sys
import numpy as np

Sgm = float(sys.argv[2])
lmbd = float(sys.argv[1])

X = np.loadtxt(sys.argv[3], delimiter=',')
Y = np.loadtxt(sys.argv[4], delimiter=',')
X0 = np.loadtxt(sys.argv[5], delimiter=',')

def RidgeRegression(X, Y, Sgm, lmbd):
    
    XTransp = X.T
    Covar = np.dot(XTransp,X)
    Reg = lmbd*np.identity(Covar.shape[1])
    Reg[-1,-1] = 0
    W_rr = np.dot(np.dot(np.linalg.inv(Reg+Covar),XTransp),Y)
        
    return W_rr

W = RidgeRegression(X, Y, Sgm, lmbd)

def ActiveRegression(X, X0, Sgm, lmbd):
    
    # compute the prior covariance
    XTransp = X.T
    Covariates = np.dot(XTransp,X)
    Reg = lmbd*np.identity(Covariates.shape[1])
    Reg[-1,-1] = 0

    InvPostCovar = Reg+((1/(Sgm**2.0)) * Covariates)
    
    X0_new = X0
    NoTestObj = X0.shape[0]
    PickedTests = np.zeros((X0.shape))
    for i in range(NoTestObj):
        
        PredCovar = (Sgm**2.0) + np.dot(np.dot(X0_new,np.linalg.inv(InvPostCovar)),X0_new.T)
        PredCovar = PredCovar.diagonal()
        
        MaxIndx = np.argmax(PredCovar)

        PickedX0 = X0_new[MaxIndx,:]

        InvPostCovar = InvPostCovar + ((1/(Sgm**2.0)) * np.outer(PickedX0, PickedX0))     
        
        X0_new = np.delete(X0_new, MaxIndx, 0)
        
        PickedTests[i,:] = PickedX0
        
    return PickedTests

PickedTests = ActiveRegression(X, X0, Sgm, lmbd)

#print PickedTests

MaxIndx = []
for i in range(PickedTests.shape[0]):
    for j in range(X0.shape[0]):
        if np.all(X0[j,:]==PickedTests[i,:]):
            MaxIndx.append(j)

np.savetxt('wRR_'+str(lmbd)+'.csv', W, fmt='%5.3f', delimiter=',')
np.savetxt('active_'+str(lmbd)+'_'+str(Sgm)+'.csv', MaxIndx[0:10], fmt='%d', newline=',')