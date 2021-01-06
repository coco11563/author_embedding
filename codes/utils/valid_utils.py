from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def evaluator(embs, label):
    estimator = KMeans(n_clusters=4)
    estimator.fit(embs)
    label_pred = estimator.labels_
    NMI = metrics.normalized_mutual_info_score(label,label_pred)
    ARI = metrics.adjusted_rand_score(label,label_pred)
    print('NMI:%.4f'%NMI)
    print('ARI:%.4f'%ARI)

    scaler=preprocessing.StandardScaler()
    X=scaler.fit_transform(embs)
    X=np.mat(X)
    train_X,test_X,train_y,test_y=train_test_split(X,label,test_size=0.6)
    model=LogisticRegression()
    model.fit(train_X,train_y)
    pred_y=model.predict(test_X)
    MICRO_F1 = f1_score(test_y,pred_y,average='micro')
    MACRO_F1 = f1_score(test_y,pred_y,average='macro')
    print('Micro-F1:%.4f'%MICRO_F1)
    print('Macro-F1:%.4f'%MACRO_F1)
    return NMI, ARI, MICRO_F1, MACRO_F1


if __name__ == '__main__':
    a = np.random.randn(128,10)
    label = np.random.randint(low=0, high=2, size=(128), dtype=int)
    print(a)
    print(label)
    evaluator(a, label)