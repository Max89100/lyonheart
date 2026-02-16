import deeplearning_library as dl
import numpy as np
import os


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



    

if __name__ == "__main__":
       test_MNIST()

