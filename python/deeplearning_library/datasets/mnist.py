from ..data import Dataset
# from ..deeplearning_library import 
import os

class MNIST(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.classes, self.class_to_idx = self._find_classes(self.data_path)
        self.samples = self._make_dataset(self.data_path,self.class_to_idx)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        raise NotImplementedError
    
    def _find_classes(self,path):
        classes = [d.name for d in os.scandir(path) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def _make_dataset(self,dir,class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir,target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root,fname)
                    item = (path,class_to_idx[target])
                    images.append(item)
        return images
                
