import numpy as np
class Kernel(object):
    '''
        Kernels for making Gram matrix.\n
        Members:\n
            self.sigma\n
            self.p\n
            self.gamma\n
            self.zeta\n
        Method:\n
            self.Linear()\n
            self.Gaussian()\n
            self.Poly()\n
            self.sigmoid()\n
        Parameters:\n
            X,Y\t               Two parts of the data.\n
            types='linear'\t     The type of kernels.\n
            sigma\n
            gamma\n
            p\n
            zeta\n
    '''
    def __init__(self,X,Y,types='linear',sigma=1,gamma=1,p=1,zeta=1) -> None:
        self.sigma,self.gamma,self.p,self.zeta=sigma,gamma,p,zeta
        if types=='linear':
            self.Linear(X,Y)
        elif types=='rbf':
            self.Gaussian(X,Y)
        elif types=='poly':
            self.Poly(X,Y)

    def Linear(self,X,Y):
        '''Calculate the X,Y dot product.'''
        self.K = X @ Y.T
    
    def Gaussian(self,X,Y):
        '''The expression is K(i,j)=exp(-L2norm(i,j)^2/2*sigma^2)'''
        X_shape=np.shape(X)[0]
        Y_shape=np.shape(Y)[0]
        self.K=np.zeros((X_shape,Y_shape),dtype=np.float64)
        for i in range(X_shape):
            for j in range(Y_shape):
                temp=(X[i]-Y[j]) @ (X[i]-Y[j]).T
                temp=temp.sum()
                self.K[i][j]=np.exp(-temp/(2*(self.sigma**2)),dtype=np.float64)
    
    def Poly(self,X,Y):
        '''The expression is K(i,j)=(gamma*ij+zeta)^p'''
        self.Linear(X,Y)
        self.K=np.power(self.gamma*self.K+self.zeta,self.p)