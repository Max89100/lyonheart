from ..data import Dataset
import os
import numpy as np

class MNIST(Dataset):
    def __init__(self, data_path, train=True, one_hot=False):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.one_hot = one_hot
        self.images, self.labels = self._make_dataset(self.data_path)
        self.num_samples = len(self.images)
        self.num_features = len(self.images[1])
        self.num_classes = len(self.labels[1])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
       return self.images[idx], self.labels[idx]
    
    def one_hot_encoding(self,array):
        num_classes = 10
        n = len(array)
        target = np.zeros((n, num_classes), dtype=np.float32)
        target[np.arange(n), array] = 1.0
        return target
    
    def _make_dataset(self,path):
        data_path = path
        files = {
            True:("train-images","train-labels"),
            False:("test-images","test-labels")
        }
        img_name, label_name = files[self.train]
        images_path = os.path.join(data_path, img_name)
        labels_path = os.path.join(data_path, label_name)

        with open(images_path, 'rb') as f:
            images = np.fromfile(f,dtype=np.uint8,offset=16)
        
        with open(labels_path, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8, offset=8)

        if(self.one_hot):
            labels = self.one_hot_encoding(labels)
        
        return images.reshape(-1, 784).astype(np.float32) / 255.0, labels

                
