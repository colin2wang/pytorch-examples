import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dsets
from torch.utils.data import DataLoader

from knn_classify import kNN_classify

batch_size = 100

mnist_train_dataset = dsets.MNIST(root="dataset", train=True, download=True)
mnist_test_dataset = dsets.MNIST(root="dataset", train=False, download=True)

train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True)

X_train = train_loader.dataset.data.numpy()
# 需要转为numpy矩阵
X_train = X_train.reshape(X_train.shape[0], 28 * 28)

# 需要reshape之后才能放入knn分类器
y_train = train_loader.dataset.targets.numpy()
X_test = test_loader.dataset.data[:100].numpy()
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
y_test = test_loader.dataset.targets[:100].numpy()
num_test = y_test.shape[0]
y_test_pred = kNN_classify(5, 'E', X_train, y_train, X_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test

print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
