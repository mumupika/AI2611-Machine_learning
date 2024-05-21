import numpy as np
import random

from mnist import Mnist
from cifar10 import Cifar10
from kernel import Kernel

class SMO(object):
    def __init__(self,X,y,C=1,iter=5000,tol=1e-2,types='linear',sigma=1,gamma=1,p=1,zeta=1):
        '''
        The main algorithm for the Sequential Minimal Optimization, Platt, 1998.\n
        Parameters:\n
            C=1                 \t The penalty weight;\n
            iter=5000           \t The maximum iteration number;\n
            tol=1e-2            \t The tolerance index;\n
            types='linear'      \t Pass to the kernel to get the Gram matrix;\n
            sigma=1             \t Pass to the kernel to rbf;\n
            gamma,p,zeta=1      \t Pass to kernel to poly;\n
        Members:\n
            self.C              \n
            self.iter           \n
            self.tol            \n    
            self.dim            \t This is the second dimension of datas.\n    
            self.sizes          \t This is the number of dataset.\n
            self.alphas         \t Lagrange multipliers. size:(1,d)\n
            self.K              \t Kernels. (Gram Matrix) size:(d,d)\n
            self.g              \t size:(1,d)\n
            self.E              \t size:(1,d)\n
            self.judges          \t size:(1,d)\n
            self.KKT_violates   \t The violation of KKT condition
            self.in_bound       \t The support vectors.\n
            self.blocks         \n
            self.bias
        '''
        '''Initialize each components'''
        self.y,self.C,self.iter,self.tol=y,C,iter,tol
        self.dim=np.shape(X)[1]
        self.sizes=np.shape(X)[0]       
        self.alphas=np.zeros((1,self.sizes),dtype=np.float64)
        self.bias=0.
        self.Kernel=Kernel(X,X,types=types,sigma=sigma,gamma=gamma,p=p,zeta=zeta)
        self.K=self.Kernel.K
        for count in range(iter):
            self.blocks=self.alphas*self.y
            self.g=self.blocks @ self.K + self.bias
            self.judges=self.g * self.y
            self.E=self.g-self.y
            self.obj_func=0.5 * self.blocks @ self.K @ self.blocks.T - self.alphas.sum()

            self.KKT_violates=[]
            self.KKT_in_bound_violates=[]
            #self.in_bound_index=[]

            self.choose_alpha1()

            if self.KKT_violates.max()<tol:
                print("stop at iter:",count,"The max KKT violate is:",self.KKT_violates.max())
                break
            self.choose_alpha2()
            self.calculate()

            if count % (iter//5) == 0:
                print(f"Object function:{self.obj_func},most KKT_violates:{self.KKT_violates.max()},index:{self.KKT_violates.argmax()},choosing alpha index:{self.alpha_1_index},{self.alpha_2_index}")

        self.blocks=self.alphas*self.y
        self.obj_func=0.5 * self.alphas @ self.K @ self.alphas.T - self.alphas.sum()
    def cut_alpha2(self,alpha_1_old,alpha_2_old,alpha_2_new):
        if self.y[0][self.alpha_1_index]==self.y[0][self.alpha_2_index]:
            L,H=max(0,alpha_2_old+alpha_1_old-self.C),min(self.C,alpha_2_old+alpha_1_old)
        else:
            L,H=max(0,alpha_2_old-alpha_1_old),min(self.C,self.C+alpha_2_old-alpha_1_old)
        
        if alpha_2_new > H:
            self.alpha_2=H
        elif alpha_2_new < L:
            self.alpha_2=L
        else:
            self.alpha_2=alpha_2_new

    def compute_bias(self,alpha_1_old,alpha_2_old):
        if self.alpha_1>0 and self.alpha_1<self.C:
            self.bias=-self.E[0][self.alpha_1_index]-self.y[0][self.alpha_1_index]*self.K[self.alpha_1_index][self.alpha_1_index]*(self.alpha_1-alpha_1_old)-self.y[0][self.alpha_2_index]*self.K[self.alpha_2_index][self.alpha_1_index]*(self.alpha_2-alpha_2_old)+self.bias
        elif self.alpha_2>0 and self.alpha_2<self.C:
            self.bias=-self.E[0][self.alpha_2_index]-self.y[0][self.alpha_1_index]*self.K[self.alpha_1_index][self.alpha_2_index]*(self.alpha_1-alpha_1_old)-self.y[0][self.alpha_2_index]*self.K[self.alpha_2_index][self.alpha_2_index]*(self.alpha_2-alpha_2_old)+self.bias
        else:
            b1=-self.E[0][self.alpha_1_index]-self.y[0][self.alpha_1_index]*self.K[self.alpha_1_index][self.alpha_1_index]*(self.alpha_1-alpha_1_old)-self.y[0][self.alpha_2_index]*self.K[self.alpha_2_index][self.alpha_1_index]*(self.alpha_2-alpha_2_old)+self.bias
            b2=-self.E[0][self.alpha_2_index]-self.y[0][self.alpha_1_index]*self.K[self.alpha_1_index][self.alpha_2_index]*(self.alpha_1-alpha_1_old)-self.y[0][self.alpha_2_index]*self.K[self.alpha_2_index][self.alpha_2_index]*(self.alpha_2-alpha_2_old)+self.bias
            self.b=(b1+b2)/2


    def calculate(self):
        alpha_1_old=self.alpha_1
        alpha_2_old=self.alpha_2
        
        ita=self.K[self.alpha_1_index][self.alpha_1_index]+self.K[self.alpha_2_index][self.alpha_2_index]-self.K[self.alpha_1_index][self.alpha_2_index]-self.K[self.alpha_2_index][self.alpha_1_index]+1e-10
        alpha_2_new=alpha_2_old+self.y[0][self.alpha_2_index]*(self.E[0][self.alpha_1_index]-self.E[0][self.alpha_2_index])/ita

        self.cut_alpha2(alpha_1_old,alpha_2_old,alpha_2_new)

        self.alpha_1=alpha_1_old+self.y[0][self.alpha_1_index]*self.y[0][self.alpha_2_index]*(alpha_2_old-self.alpha_2)
        if self.alpha_1<0: self.alpha_1=0
        elif self.alpha_1>self.C: self.alpha_1=self.C

        self.alphas[0][self.alpha_1_index],self.alphas[0][self.alpha_2_index]=self.alpha_1,self.alpha_2

        self.compute_bias(alpha_1_old,alpha_2_old)
        
    
    def choose_alpha1(self):
        for i in range(self.sizes):
            if self.alphas[0][i]==0 and self.judges[0][i]>=1:
                self.KKT_violates.append(0.)
            elif self.alphas[0][i]==self.C and self.judges[0][i]<=1:
                self.KKT_violates.append(0.)
            else:
                self.KKT_violates.append(abs(self.judges[0][i]-1))
                if self.alphas[0][i] > 0 and self.alphas[0][i] < self.C:
                    self.KKT_in_bound_violates.append(abs(self.judges[0][i]-1))

        self.KKT_violates=np.array(self.KKT_violates,dtype=np.float64)
        self.KKT_in_bound_violates=np.array(self.KKT_in_bound_violates,dtype=np.float64)
        # Finding the most violates.
        if np.random.random()<0.99:
            if self.KKT_in_bound_violates.size != 0:
                in_bound_max_violate=self.KKT_in_bound_violates.max()
                self.alpha_1_index=np.argwhere(self.KKT_violates==in_bound_max_violate)[0].sum()
                self.alpha_1=self.alphas[0][self.alpha_1_index]
            else:
                self.alpha_1_index=np.argmax(self.KKT_violates)
                self.alpha_1=self.alphas[0][self.alpha_1_index]
        else:
            self.alpha_1_index=np.argmax(self.KKT_violates)
            self.alpha_1=self.alphas[0][self.alpha_1_index]

    def choose_alpha2(self):
        self.alpha_2_index=self.alpha_1_index
        while self.alpha_2_index==self.alpha_1_index:
            self.alpha_2_index=random.randint(0,self.sizes-1)
        self.alpha_2=self.alphas[0][self.alpha_2_index]