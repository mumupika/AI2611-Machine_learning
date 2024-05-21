import numpy as np
from mnist import Mnist
from cifar10 import Cifar10
from classification import SVM_45_classes
if __name__=='__main__':
    res=SVM_45_classes(class_size=100,dataset='Cifar10',C=0.25,iter=5000,tol=1e-2,types='poly',p=8,zeta=2,gamma=2)