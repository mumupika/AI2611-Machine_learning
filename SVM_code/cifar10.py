import cv2
import numpy as np
from PIL import Image
import pickle
import os

from dataset import Dataset
class Cifar10(Dataset):
    category='Cifar10'
    train_image={}          
    test_image={}           
    train_data={}           
    test_data={} 
    def __init__(self,require_norms=True,hog=True) -> None:
        
        super(Dataset,self).__init__()
        Dataset.__init__(self)

        self.winSize = (32,32)
        self.blockSize = (16,16)
        self.blockStride = (4,4)
        self.cellSize=(8,8)         #(8,8)
        self.nbins=9
        if os.path.exists('Cifar10/Train_Hog_data.npz') == False or os.path.exists('Cifar10/Test_Hog_data.npz') == False:
            self.Image_save()
            if hog==True:
                self.Hog()
        else:
            a,b=np.load('Cifar10/Train_Hog_data.npz'),np.load('Cifar10/Test_Hog_data.npz')
            self.train_data['imgs'],self.train_data['labels']=a['imgs'],a['labels']
            self.test_data['imgs'],self.test_data['labels']=b['imgs'],b['labels']

        print(f"The dim of the image:{np.shape(self.train_data['imgs'])[1]}")

    def Hog(self):
        hog=cv2.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins)

        if os.path.exists('Cifar10/real_images') == True:
            train_images,train_labels=[],[]
            for i in range(10000):
                label=self.train_image['labels'][i]
                img=cv2.imread(f'Cifar10/real_images/train_No.{i}_label={label}.jpg',cv2.IMREAD_GRAYSCALE)
                img=hog.compute(img)
                train_images.append(img)
                train_labels.append(label)
            test_images,test_labels=[],[]
            for i in range(10000):
                label=self.test_image['labels'][i]
                img=cv2.imread(f'Cifar10/real_images/test_No.{i}_label={label}.jpg',cv2.IMREAD_GRAYSCALE)
                img=hog.compute(img)
                test_images.append(img)
                test_labels.append(label)
            
        self.train_data['imgs'],self.train_data['labels']=np.array(train_images,dtype=np.float64),np.array(train_labels,dtype=np.int64)
        self.test_data['imgs'],self.test_data['labels']=np.array(test_images,dtype=np.float64),np.array(test_labels,dtype=np.int64)

        if os.path.exists('Cifar10/Train_Hog_data.npz') == False:
            np.savez('Cifar10/Train_Hog_data.npz',imgs=self.train_data['imgs'],labels=self.train_data['labels'])
        if os.path.exists('Cifar10/Test_Hog_data.npz') == False:
            np.savez('Cifar10/Test_Hog_data.npz',imgs=self.test_data['imgs'],labels=self.test_data['labels'])


    def Image_save(self):
        with open(self.train_dict_path,'rb') as f:
            train_dict=pickle.load(f,encoding='bytes')
        with open(self.test_dict_path,'rb') as f:
            test_dict=pickle.load(f,encoding='bytes')
        
        self.train_image['img'],self.train_image['labels']=train_dict[b'data'].reshape(-1,3,32,32),train_dict[b'labels']
        self.test_image['img'],self.test_image['labels']=test_dict[b'data'].reshape(-1,3,32,32),test_dict[b'labels']

       

        if os.path.exists('Cifar10/real_images') == False:
            os.mkdir('Cifar10/real_images')
            for i in range(10000):
                image=self.train_image['img'][i]
                R,G,B=Image.fromarray(image[0]),Image.fromarray(image[1]),Image.fromarray(image[2])
                real_image=Image.merge("RGB",(R,G,B))
                label=self.train_image['labels'][i]
                real_image.save(f'Cifar10/real_images/train_No.{i}_label={label}.jpg')
            
            for i in range(10000):
                image=self.test_image['img'][i]
                R,G,B=Image.fromarray(image[0]),Image.fromarray(image[1]),Image.fromarray(image[2])
                real_image=Image.merge("RGB",(R,G,B))
                label=self.test_image['labels'][i]
                real_image.save(f'Cifar10/real_images/test_No.{i}_label={label}.jpg')
        