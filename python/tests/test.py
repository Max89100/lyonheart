import deeplearning_library as dl
from deeplearning_library.data.datasets import MNIST
from deeplearning_library import DataLoader, Sequential, Linear, ReLU,Softmax, SGD, MSELoss, CrossEntropyLoss, LogSoftmax, CoreTensor, InitMethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_layers():
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    x = dl.CoreTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    for i in range(3):
        h = l1.forward(x).relu()
        y = l2.forward(h).sigmoid()
        res = y.to_numpy()
        print("Prédictions XOR (Avant entraînement) :")
        print(res)

def test_MLP_XOR():
    x = dl.CoreTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
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
        loss = dl.LossFunction.mse(y_pred,dl.CoreTensor(np.array([[0,0],[1,1],[1,1],[0,0]], dtype=np.float32)))
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
    x = core.CoreTensor(np.array([[2.0,1.0,0.1]], dtype=np.float32))
    softmax = x.softmax()
    print(softmax.to_numpy())
    res = core.LossFunction.cross_entropy(softmax,core.CoreTensor(np.array([[1.0,0.0,0.0]], dtype=np.float32)))
    print(res.to_numpy())

def test_MNIST():

    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    evaluation = MNIST("../../data/mnist", train=False,one_hot=True)
    dataloader = DataLoader(dataset, batch_size=64,shuffle=True)
    print(dataset.num_features)
    print(dataset.num_classes)

    # Modèle
    l1 = core.Linear(dataset.num_features,128, core.InitMethod.Kaiming)
    l2 = core.Linear(128,dataset.num_classes, core.InitMethod.Xavier)

    # Entraînement
    epoch = 2
    for n in tqdm(range(epoch)):
        valids = 0
        for i, (images, targets) in enumerate(dataloader):
            x_tensor = core.CoreTensor(images)
            y_tensor = core.CoreTensor(targets)
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
        x_tensor = core.CoreTensor(images)
        y_tensor = core.CoreTensor(targets)
        f = l1.forward(x_tensor).relu()
        g = l2.forward(f).softmax()
        loss = core.LossFunction.cross_entropy(g,y_tensor)
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

def test_MNIST_enhanced():
    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    model = Sequential([Linear(784,128),ReLU(),Linear(128,10,InitMethod.Xavier)])
    criterion = LogSoftmax()
    optimizer = SGD(model.parameters(),0.01)
    
    for n in tqdm(range(2)):
        valids = 0
        for i, (images, labels) in enumerate(dataloader):
            y_pred = model(CoreTensor(images))
            y_target = CoreTensor(labels)
            loss = criterion(y_pred,y_target)
            model.backward(loss)
            optimizer.step()
            preds = np.argmax(y_pred.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
            targets = np.argmax(y_target.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
            valids = valids + np.sum(preds == targets)
        accuracy = np.divide(valids,dataset.num_samples)
        print(accuracy)
        print(loss.to_numpy())

def test_overloading_operators():
    x = core.CoreTensor(np.array([[1,2]], dtype=np.float32))
    y = core.CoreTensor(np.array([[3,4]],dtype=np.float32))
    w = core.CoreTensor(np.array([[3,4], [5,6]],dtype=np.float32))
    add = x.add(y)
    iadd = x
    iadd += y
    sub = x - y
    mul = x * y
    truediv = x / y
    matmul = x@w
    neg = -w

    print(add, iadd, sub,mul,truediv,matmul,neg)

def test_Module():
    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    model = Sequential([Linear(784,128),ReLU(),Sequential([Linear(128,64), ReLU(), Linear(64,10, InitMethod.Xavier)])])
    criterion = LogSoftmax()
    optimizer = SGD(model.parameters(),0.01)

    for n in tqdm(range(2)):
        valids = 0
        for i, (images, labels) in enumerate(dataloader):
            y_pred = model(CoreTensor(images))
            y_target = CoreTensor(labels)
            loss = criterion(y_pred,y_target)
            model.backward(loss)
            optimizer.step()
            preds = np.argmax(y_pred.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
            targets = np.argmax(y_target.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
            valids = valids + np.sum(preds == targets)
        accuracy = np.divide(valids,dataset.num_samples)
        print(accuracy)
        print(loss.to_numpy())

def test_save_load():
    dataset = MNIST("../../data/mnist", train=True,one_hot=True)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    model = Sequential([Linear(784,128),ReLU(),Linear(128,10,InitMethod.Xavier)])
    criterion = LogSoftmax()
    optimizer = SGD(model.parameters(),0.01)
    
    for n in tqdm(range(2)):
        valids = 0
        for i, (images, labels) in enumerate(dataloader):
            y_pred = model(CoreTensor(images))
            y_target = CoreTensor(labels)
            loss = criterion(y_pred,y_target)
            model.backward(loss)
            optimizer.step()
            preds = np.argmax(y_pred.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
            targets = np.argmax(y_target.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
            valids = valids + np.sum(preds == targets)
        accuracy = np.divide(valids,dataset.num_samples)
        print(accuracy)
        print(loss.to_numpy())
    
    dl.save(model,"model.pkl")
    model2 = Sequential([Linear(784,128),ReLU(),Linear(128,10,InitMethod.Xavier)])
    state = dl.load("model.pkl")
    model2.load_state_dict(state)
    #test
    valids = 0
    for i, (images, labels) in enumerate(dataloader):
        y_pred = model2(CoreTensor(images))
        y_target = CoreTensor(labels)
        preds = np.argmax(y_pred.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
        targets = np.argmax(y_target.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
        valids = valids + np.sum(preds == targets)
    accuracy = np.divide(valids,dataset.num_samples)
    print(accuracy)



if __name__ == "__main__":
    test_save_load()
    
    
    
    

    
    

    


