import operator

import numpy as np
import matplotlib.pyplot as plt


##给出训练数据以及对应的类别
def createDataSet():
    _group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5], [1.1, 1.0], [0.5, 1.5]])
    _labels = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    return _group, _labels


def kNN_classify(k, dis, X_train, x_train, Y_test):
    assert dis == 'E' or dis == 'M', 'dis must Eor M，E代表欧式距离，M代表曼哈顿距离'
    # 测试样本的数量
    num_test = Y_test.shape[0]

    label_list = []
    '''
    使用曼哈顿公式作为距离度量
    '''
    if dis == 'M':
        for i in range(num_test):
            # 实现欧式距离公式
            tmp1 = np.sum(np.absolute((X_train - np.tile(Y_test[i], (X_train.shape[0], 1)))), axis=1)
            distances = np.sqrt(tmp1)
            # 距离由小到大进行排序，并返回index值
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]  # 选取前k个距离
            class_count = {}
            for j in top_k:  # 统计每个类别的个数
                class_count[x_train[j]] = class_count.get(x_train[j], 0) + 1
            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sorted_class_count[0][0])
        return np.array(label_list)
    '''
    使用欧拉公式作为距离度量
    '''
    if dis == 'E':
        for i in range(num_test):
            # 实现欧式距离公式
            distances = np.sqrt(np.sum(((X_train - np.tile(Y_test[i], (X_train.shape[0], 1))) ** 2), axis=1))
            # 距离由小到大进行排序，并返回index值
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]  # 选取前k个距离
            class_count = {}
            for j in top_k:  # 统计每个类别的个数
                class_count[x_train[j]] = class_count.get(x_train[j], 0) + 1
            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sorted_class_count[0][0])
        return np.array(label_list)


if __name__ == '__main__':
    group, labels = createDataSet()

    y_test_group = np.array([[1.0, 2.1], [0.4, 2.0]])
    y_test_labels = kNN_classify(1, 'M', group, labels, y_test_group)
    print(y_test_labels)  # 打印输出['A' 'B']，和我们的判断是相同的

    # 对于类别为A的数据集我们使用红色六角形表示
    plt.scatter(group[labels == 'A', 0], group[labels == 'A', 1], color='r', marker='*')

    # 对于类别为B的数据集我们使用绿色十字形表示
    plt.scatter(group[labels == 'B', 0], group[labels == 'B', 1], color='g', marker='+')

    # 对于类别为A的数据集我们使用红色六角形表示
    plt.scatter(y_test_group[y_test_labels == 'A', 0], y_test_group[y_test_labels == 'A', 1], color='y', marker='*')

    # 对于类别为B的数据集我们使用绿色十字形表示
    plt.scatter(y_test_group[y_test_labels == 'B', 0], y_test_group[y_test_labels == 'B', 1], color='y', marker='+')

    plt.show()
