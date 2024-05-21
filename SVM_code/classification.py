import numpy as np

from mnist import Mnist
from cifar10 import Cifar10
from smo import SMO
from kernel import Kernel

class SVM_45_classes(object):
    def __init__(self,class_size=100,dataset='Mnist',C=1,iter=1000,tol=1e-2,types='linear',sigma=1,gamma=1,p=1,zeta=1) -> None:
        self.class_size,self.dataset=class_size,dataset
        self.C,self.iter,self.tol,self.types,self.sigma,self.gamma,self.p,self.zeta=C,iter,tol,types,sigma,gamma,p,zeta
        if dataset=='Mnist':
            self.data=Mnist(True)
            self.train_data,self.train_labels=self.data.train_data['imgs'],self.data.train_data['labels']
            self.test_data,self.test_labels=self.data.test_data['imgs'],self.data.test_data['labels']
            self.dim=28*28
        elif dataset=='Cifar10':
            self.data=Cifar10()
            self.train_data,self.train_labels=self.data.train_data['imgs'],self.data.train_data['labels']
            self.test_data,self.test_labels=self.data.test_data['imgs'],self.data.test_data['labels']
            self.dim=np.shape(self.train_data)[1]
        
        self.dataloader()
        self.train()
        self.train_val()
        self.test()
    
    def test(self):
        dim=np.shape(self.test_data)[1]
        test_data,test_labels=self.test_data[:100].reshape(-1,self.dim),self.test_labels[:100]
        count=0
        sizes=np.shape(test_data)[0]
        predictions=np.zeros((sizes,10))
        for i in range(10):
            for j in range(i+1,10):
                train_X=self.after_train_X[count]
                train_y=self.after_train_y[count]

                Ker=Kernel(train_X,test_data,self.types,self.sigma,self.gamma,self.p,self.zeta).K
                Support_vector=self.res[count]
                pred=np.sign(Support_vector.blocks @ Ker  + Support_vector.bias + 1e-10)
                count+=1
                for k in range(sizes):
                    if pred[0][k]==1:
                        predictions[k][i] += 1
                    else:
                        predictions[k][j] += 1
        
        final=np.argmax(predictions,axis=1)
        acc=(final==test_labels).sum()/np.shape(test_labels)[0]
        print(f"Test acc:{acc*100}%.")


    def train_val(self):
        count=0
        predictions=np.zeros((10*self.class_size,10))
        for i in range(10):
            for j in range(i+1,10):
                train_X=self.after_train_X[count]
                train_y=self.after_train_y[count]

                Ker=Kernel(train_X,self.validate,self.types,self.sigma,self.gamma,self.p,self.zeta).K
                Support_vector=self.res[count]
                pred=np.sign(Support_vector.blocks @ Ker  + Support_vector.bias + 1e-10)
                count+=1
                for k in range(10*self.class_size):
                    if pred[0][k]==1:
                        predictions[k][i] += 1
                    else:
                        predictions[k][j] += 1
        
        final=np.argmax(predictions,axis=1)
        acc=(final==self.validate_labels).sum()/np.shape(self.validate_labels)[0]
        print(f"Train acc:{acc*100}%.")
                


    def dataloader(self):
        tr_data=[[],[],[],[],[],[],[],[],[],[]]
        val=[]
        val_labels=[]
        self.real_train_data=tr_data.copy()
        for i in range(np.shape(self.train_data)[0]):
            tr_data[self.train_labels[i]].append(self.train_data[i])
            val.append(self.train_data[i])
            val_labels.append(self.train_labels[i])

        for i in range(10):
            self.real_train_data[i]=tr_data[i][:self.class_size]
        
        
        self.real_train_data=np.array(self.real_train_data,dtype=np.float64).reshape(10,self.class_size,self.dim) # (10,class_size,dim)
        
        self.validate=[]
        self.validate_labels=[]

        permutation=np.random.permutation(10*self.class_size)
        for i in range(10*self.class_size):
            self.validate.append(val[permutation[i]])
            self.validate_labels.append(val_labels[permutation[i]])
        
        self.validate=np.array(self.validate,dtype=np.float64).reshape(10*self.class_size,self.dim)
        self.validate_labels=np.array(self.validate_labels,dtype=np.int64)
    def train(self):
        self.res=[]
        self.after_train_X=[]
        self.after_train_y=[]
        for i in range(10):
            for j in range(i+1,10):
                print(f"training class {i},{j}")
                train_X=[]
                train_y=[]
                for count in range(self.class_size):
                    pivot=np.random.random()
                    if pivot > 0.5:
                        train_X.append(self.real_train_data[i][count])
                        train_y.append(1)
                        train_X.append(self.real_train_data[j][count])
                        train_y.append(-1)
                    else:
                        train_X.append(self.real_train_data[j][count])
                        train_y.append(-1)
                        train_X.append(self.real_train_data[i][count])
                        train_y.append(1)
                    
                train_X=np.array(train_X,dtype=np.float64)
                train_y=np.array(train_y,dtype=np.int64).reshape(1,-1)

                self.after_train_X.append(train_X)
                self.after_train_y.append(train_y)

                result=SMO(train_X,train_y,self.C,self.iter,self.tol,self.types,self.sigma,self.gamma,self.p,self.zeta)
                self.res.append(result)
