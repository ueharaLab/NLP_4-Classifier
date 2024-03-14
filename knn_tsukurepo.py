from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.datasets import make_blobs
import seaborn as sns
import pandas as pd
import japanize_matplotlib

tsukurepo_bow= pd.read_csv('./data/tsukurepo_bow.csv', encoding='ms932', sep=',',skiprows=0)

label={'杏仁豆腐':0,'シュークリーム':1,'プリン':2}
label_inv = {v:n for n,v in label.items()}
y=[]
for k in tsukurepo_bow['keyword']:

    y.append(label[k])
        
X = tsukurepo_bow.iloc[:,4:]

#X, y = make_blobs(random_state=8, n_samples=50, n_features=2,  cluster_std=7, centers=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
k=10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
#print(y_train)
y_predict=knn.predict(X_test)

print('predict accuracy :{0} no.of test data :{1}'.format(knn.score(X_test,y_test),len(X_test)))
for p, t in zip(y_predict,y_test):

    print('predict:{0}  true value:{1}'.format(label_inv[p],label_inv[t]) )



