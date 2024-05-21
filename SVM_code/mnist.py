import numpy as np
from PIL import Image
import os

from dataset import Dataset


class Mnist(Dataset):
    '''
    The following is the Mnist dataset.\n
    Members:\n
        train(test)_image;\t        Dict,{'imgs','labels'},uint8\n
        train(test)_data;\t         Dict,{'imgs','labels'},float64\n
    '''

    category='Mnist'
    train_image={}          
    test_image={}           
    train_data={}           
    test_data={}            

    def __init__(self,require_norms=True) -> None:
        super(Dataset,self).__init__()
        Dataset.__init__(self)

        if os.path.exists('Mnist/Train_images_and_labels_array.npz') == False:
            self.get_train_image()
        else:
            self.train_image=np.load('Mnist/Train_images_and_labels_array.npz')
        
        if os.path.exists('Mnist/Test_images_and_labels_array.npz') == False:
            self.get_test_image()
        else:
            self.test_image=np.load('Mnist/Test_images_and_labels_array.npz')
        
        self.get_train_and_test_data(require_norms)
        # self.display(type='train',num=999)

    def get_train_and_test_data(self,require_norms):
        '''
            Transfer the data into float64. \n
            If require_norms=True, then normalize to 0 and 1.
        '''
        if os.path.exists('Mnist/train_datas.npz')==False or os.path.exists('Mnist/test_datas.npz')==False:
            train_data,test_data=self.train_image,self.test_image
            if require_norms==True:
                self.train_data['imgs']=np.array(train_data['imgs'],dtype=np.float64)/255
                self.test_data['imgs']=np.array(test_data['imgs'],dtype=np.float64)/255
                self.train_data['labels']=train_data['labels']
                self.test_data['labels']=test_data['labels']
            else:
                self.train_data['imgs']=np.array(train_data['imgs'],dtype=np.float64)/255
                self.test_data['imgs']=np.array(test_data['imgs'],dtype=np.float64)/255
                self.train_data['labels']=train_data['labels']
                self.test_data['labels']=test_data['labels']

            np.savez('Mnist/train_datas.npz',imgs=self.train_data['imgs'],labels=self.train_data['labels'])
            np.savez('Mnist/test_datas.npz',imgs=self.test_data['imgs'],labels=self.test_data['labels'])
        
        else:
            a,b=np.load('Mnist/train_datas.npz'),np.load('Mnist/test_datas.npz')
            self.train_data['imgs'],self.train_data['labels']=a['imgs'],a['labels']
            self.test_data['imgs'],self.test_data['labels']=b['imgs'],b['labels']

    def get_train_image(self):
        with open(self.train_img_path,'rb') as f:
            tr_img=f.read()
        with open(self.train_labels_path,'rb') as f:
            tr_labels=f.read()
        
        images=[]
        labels=[]
        for i in range(60000):
            img=[item for item in tr_img[ (16+28*28*i) : (16+28*28*(i+1)) ]]
            img=np.array(img,dtype=np.uint8).reshape(28,28)
            images.append(img)

        for i in range(60000):
            label=int.from_bytes(tr_labels[8+i:8+i+1],'big')
            labels.append(label)
        
        images=np.array(images,dtype=np.uint8)
        labels=np.array(labels,dtype=np.uint8)
        self.train_image['imgs'],self.train_image['labels']=images,labels
        np.savez('Mnist/Train_images_and_labels_array.npz',imgs=self.train_image['imgs'],labels=self.train_image['labels'])


    def get_test_image(self):
        with open(self.test_img_path,'rb') as f:
            t_img=f.read()
        with open(self.test_labels_path,'rb') as f:
            t_labels=f.read()

        images,labels=[],[]
        for i in range(10000):
            img=[item for item in t_img[ (16+28*28*i) : (16+28*28*(i+1)) ]]
            img=np.array(img,dtype=np.uint8).reshape(28,28)
            images.append(img)
        for i in range(10000):
            label=int.from_bytes(t_labels[8+i:8+i+1],'big')
            labels.append(label)
        
        images=np.array(images,dtype=np.uint8)
        labels=np.array(labels,dtype=np.uint8)
        self.test_image['imgs'],self.test_image['labels']=images,labels
        np.savez('Mnist/Test_images_and_labels_array.npz',imgs=self.test_image['imgs'],labels=self.test_image['labels'])

    def display(self,type,num):
        if type=='train':
            imgs=self.train_image['imgs']
            labels=self.train_image['labels']
        elif type=='test':
            imgs=self.test_image['imgs']
            labels=self.test_image['labels']
        real_image=Image.fromarray(imgs[num])
        real_image.show()
        print(labels[num])
