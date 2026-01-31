import deeplearning_library as dl
import numpy as np

  
def test_layers():
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Avant entraînement) :")
    print(res)

def test_MLP_XOR():
    x = dl.GpuTensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
    l1 = dl.Linear(2,4)
    l2 = dl.Linear(4, 1)
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Avant entraînement) :")
    print(res)
    for i in range(10) :
        #forward pass
        h = l1.forward(x).relu()
        y_pred = l2.forward(h).sigmoid()
        loss = dl.LossFunction.mse(y_pred,dl.GpuTensor(np.array([[0,0],[1,1],[1,1],[0,0]], dtype=np.float32)))
        print(loss)
        #backward pass
        l1.update(0.02,loss)
        l2.update(0.02,loss)

    #Evaluation
    h = l1.forward(x).relu()
    y = l2.forward(h).sigmoid()
    res = y.to_numpy()
    print("Prédictions XOR (Après entraînement) :")
    print(res)
    


if __name__ == "__main__":
       test_MLP_XOR()


