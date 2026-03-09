# 🦁 LyonHeart : Hybrid Deep Learning Framework

LyonHeart est un framework de Deep Learning hybride conçu pour allier l'expressivité de **Python** à la performance et la sécurité mémoire de **Rust**. Ce projet repose sur une architecture où le noyau de calcul et l'autodiff sont gérés par un backend Rust, exposé via PyO3.

## 🚀 Installation

Le projet est distribué sous forme de `wheel` Python pré-compilée. Pour l'installer dans votre environnement, utilisez `pip` :

```bash
pip install lyonheart-0.1.0-cp312-cp312-win_amd64.whl
```

## Test
Placez le dossier data/mnist à l'emplacement du code (attention à la manière dont vous réalisez l'extraction !)

```python
import lyonheart as lh
from lyonheart import *

dataset = lh.data.datasets.MNIST("data/mnist", train=True,one_hot=True)
train_loader = DataLoader(dataset,64,True)
test_loader = DataLoader(dataset,64,False)
model = Sequential([Linear(784,128), ReLU(), Linear(128,10)])
criterion = lh.losses.LogSoftmax()
optimizer = lh.optim.SGD(model.parameters(),lr = 0.1)
trainer = Trainer(model,optimizer,[Accuracy()])
trainer.train(10,train_loader,criterion)
trainer.evaluate(test_loader)
```

## Code source
https://github.com/Max89100/lyonheart/tree/dev