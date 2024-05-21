class Dataset(object):
    '''
    base class, get different dataset path.
    '''
    category=''
    def __init__(self) -> None:
        if self.category=='Mnist':
            self.train_img_path='Mnist/train-images.idx3-ubyte'
            self.train_labels_path='Mnist/train-labels.idx1-ubyte'
            self.test_img_path='Mnist/t10k-images.idx3-ubyte'
            self.test_labels_path='Mnist/t10k-labels.idx1-ubyte'
        elif self.category=='Cifar10':
            self.train_dict_path='Cifar10/data_batch_1'
            self.test_dict_path='Cifar10/test_batch'

