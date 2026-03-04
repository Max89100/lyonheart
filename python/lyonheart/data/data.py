import numpy as np

# Classe Dataset abstraite
class Dataset:
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self,idx):
        raise NotImplementedError
    
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __iter__(self): 
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))
        
        for i in range(len(self)):
            batch_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            features, targets = zip(*batch)
            yield np.stack(features, axis=0), np.stack(targets,axis=0)