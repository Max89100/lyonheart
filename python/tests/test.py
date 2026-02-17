import deeplearning_library as dl
import numpy as np
import os
from deeplearning_library.datasets import IntelDataset
from deeplearning_library import DataLoader
import matplotlib.pyplot as plt


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


def load_mnist(images, labels):
    # On récupère le dossier où se trouve le fichier test.py (le script actuel)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # On remonte de deux niveaux pour atteindre la racine du projet, puis on descend dans /data
    # Cela correspond à ton "../../data" mais de manière robuste
    data_dir = os.path.join(current_dir, "..", "..", "data")
    
    images_path = os.path.join(data_dir, images)
    labels_path = os.path.join(data_dir, labels)

    print(f"Tentative d'ouverture de : {os.path.abspath(images_path)}")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Fichier introuvable à l'adresse : {os.path.abspath(images_path)}")

    with open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    
    with open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    return images.reshape(-1, 784).astype(np.float32) / 255.0, labels.astype(np.int64)

def test_MNIST():
    x_train, y_train = load_mnist("train-images", "train-labels")
    #x_test, y_test = load_mnist("test-images", "test-labels")
    print(x_train.shape)
    print(y_train.shape)

    # 1. Préparation
    first_image = x_train[0:1] # On prend la première ligne, garde le format [1, 784]
    target = np.zeros((1, 10))
    target[0, y_train[0]] = 1.0 # Petit One-hot manuel pour le test

    # 2. Conversion vers ton framework
    x_gpu = dl.GpuTensor(first_image)
    y_gpu = dl.GpuTensor(np.array(target,dtype=np.float32))

    # modèle
    l1 = dl.Linear(784,128,dl.InitMethod.Kaiming)
    l2 = dl.Linear(128,10,dl.InitMethod.Xavier)

    h = l1.forward(x_gpu).relu()
    y_pred = l2.forward(h).softmax()
    loss = dl.LossFunction.cross_entropy(y_pred,y_gpu)
    print(y_pred.to_numpy())
    print(y_gpu.to_numpy())
    print(loss.to_numpy())

def one_hot_encoding(array):
    target = np.zeros((len(array),10))
    for i in range(len(array)):
        target[i,array[i]] = 1.0
    return target

def real_mnist_test():
    x_train, y_train = load_mnist("train-images", "train-labels")
    x_test, y_test = load_mnist("test-images", "test-labels")
    epoch = 2
    batch_size = 64
    n_samples = len(x_train)
    input_size = len(x_train[0])
    output_size = 10

    # Modèle MLP
    l1 = dl.Linear(input_size,128, dl.InitMethod.Kaiming)
    l2 = dl.Linear(128,output_size, dl.InitMethod.Xavier)

    # Entraînement
    for n in range(0,epoch) :
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        valids = 0
        for i in range(0, n_samples, batch_size):
            stop = min(i + batch_size, n_samples)
            batch_x = x_train[i:stop]
            batch_y = one_hot_encoding(y_train[i:stop])
            batch_x_tensor = dl.GpuTensor(np.array(batch_x, dtype=np.float32))
            batch_y_tensor = dl.GpuTensor(np.array(batch_y, dtype=np.float32))

            f = l1.forward(batch_x_tensor).relu()
            g = l2.forward(f).softmax()
            loss = dl.LossFunction.cross_entropy(g,batch_y_tensor)
            loss.backward()
            l1.update(0.1)
            l2.update(0.1)
            preds = np.argmax(g.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
            targets = np.argmax(batch_y_tensor.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
            valids = valids + np.sum(preds == targets)
        accuracy = np.divide(valids,n_samples)
        print(accuracy)

    
    # Evaluation
    n_test_samples = len(x_test)
    valids = 0
    for i in range(0,n_test_samples,batch_size):
        stop = min(i + batch_size, n_test_samples)
        batch_x = x_test[i:stop]
        batch_y = one_hot_encoding(y_test[i:stop])
        batch_x_tensor = dl.GpuTensor(np.array(batch_x, dtype=np.float32))
        batch_y_tensor = dl.GpuTensor(np.array(batch_y, dtype=np.float32))
        f = l1.forward(batch_x_tensor).relu()
        g = l2.forward(f).softmax()
        loss = dl.LossFunction.cross_entropy(g,batch_y_tensor)
        # Imaginons que 'logits' est la sortie de ton réseau [Batch, 10]
        preds = np.argmax(g.to_numpy(), axis=1) # On prend l'indice de la plus haute probabilité
        targets = np.argmax(batch_y_tensor.to_numpy(), axis=1) # On prend l'indice du 1 dans le One-Hot
        valids = valids + np.sum(preds == targets)
    accuracy = np.divide(valids,n_test_samples)
    print(accuracy)

    

if __name__ == "__main__":
    dataset = IntelDataset(data_path = '../../data/intel_dataset/seg_train')
    dataloader = DataLoader(dataset, batch_size=32,shuffle=False)
   
    for i, (images, targets) in enumerate(dataloader):
        print(f'Batch {i} images shape: {images.shape}')
        print(f'Batch {i} targets shape: {targets.shape}')
        if i == 5:
            break

