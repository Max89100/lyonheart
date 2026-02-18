import deeplearning_library as dl
from deeplearning_library import core
from deeplearning_library.datasets import IntelDataset, MNIST
from deeplearning_library import DataLoader, Sequential, Linear, ReLU,Softmax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_layers():
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    for i in range(3):
        h = l1.forward(x).relu()
        y = l2.forward(h).sigmoid()
        res = y.to_numpy()
        print("Prédictions XOR (Avant entraînement) :")
        print(res)

def test_MLP_XOR():
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    l1 = dl.Linear(2,4, dl.InitMethod.Kaiming)
    l2 = dl.Linear(4, 1, dl.InitMethod.Xavier)
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Avant entraînement) :")
    print(res)
    for i in range(5000) :
        print("epoch :",i)
        #forward pass
        h = l1.forward(x).relu()
        y_pred = l2.forward(h).sigmoid()
        loss = dl.LossFunction.mse(y_pred,dl.GpuTensor(np.array([[0,0],[1,1],[1,1],[0,0]], dtype=np.float32)))
        print(loss.to_numpy())
        #backward pass
        loss.backward()
        l1.update(0.1)
        l2.update(0.1)
        
    #Evaluation
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Après entraînement) :")
    print(res)
    
def test_softmax():
    x = dl.GpuTensor(np.array([[2.0,1.0,0.1]], dtype=np.float32))
    softmax = x.softmax()
    print(softmax.to_numpy())
    res = dl.LossFunction.cross_entropy(softmax,dl.GpuTensor(np.array([[1.0,0.0,0.0]], dtype=np.float32)))
    print(res.to_numpy())

def test_MNIST():

    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    evaluation = MNIST("../../data/mnist", train=False,one_hot=True)
    dataloader = DataLoader(dataset, batch_size=64,shuffle=True)
    print(dataset.num_features)
    print(dataset.num_classes)

    # Modèle
    l1 = dl.Linear(dataset.num_features,128, core.InitMethod.Kaiming)
    l2 = dl.Linear(128,dataset.num_classes, core.InitMethod.Xavier)

    # Entraînement
    epoch = 2
    for n in tqdm(range(epoch)):
        valids = 0
        for i, (images, targets) in enumerate(dataloader):
            x_tensor = core.GpuTensor(images)
            y_tensor = core.GpuTensor(targets)
            f = l1.forward(x_tensor).relu()
            g = l2.forward(f).softmax()
            loss = core.LossFunction.cross_entropy(g,y_tensor)
            loss.backward()
            l1.update(0.1)
            l2.update(0.1)
            preds = np.argmax(g.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
            targets = np.argmax(y_tensor.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
            valids = valids + np.sum(preds == targets)
        accuracy = np.divide(valids,len(dataset))
        print(accuracy)

    
    # Evaluation
    dataloader = DataLoader(evaluation,64,True)
    valids = 0
    for i, (images, targets) in enumerate(dataloader):
        x_tensor = dl.GpuTensor(images)
        y_tensor = dl.GpuTensor(targets)
        f = l1.forward(x_tensor).relu()
        g = l2.forward(f).softmax()
        loss = dl.LossFunction.cross_entropy(g,y_tensor)
        loss.backward()
        l1.update(0.1)
        l2.update(0.1)
        preds = np.argmax(g.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
        targets = np.argmax(y_tensor.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
        valids = valids + np.sum(preds == targets)
    accuracy = np.divide(valids,len(evaluation))
    print(accuracy)

def test_Intel_Dataset():
    dataset = IntelDataset(data_path="../../data/intel_dataset/seg_train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for i, (images, targets) in enumerate(dataloader):
        print(f'Batch {i} images shape: {images.shape}')
        print(f'Batch {i} targets shape: {targets.shape}')
        if i == 5:
            break

if __name__ == "__main__":
    
    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    model = Sequential([Linear(784,128),ReLU(),Linear(128,10),Softmax()])
    print(model.parameters())

    


