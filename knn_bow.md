# BoWにもとづく教師あり機械学習
もっとも簡単な教師あり機械学習アルゴリズム, k nearest neighbor を使ってシュークリーム、プリン、杏仁豆腐、の識別をやってみる。
[knn_tsukurepo.py](knn_tsukurepo.py)
``` python
tsukurepo_bow= pd.read_csv('tsukurepo_bow.csv', encoding='ms932', sep=',',skiprows=0)

# 教師ラベルを0,1,2の数値ラベルに変換
label={'杏仁豆腐':0,'シュークリーム':1,'プリン':2}
label_inv = {v:n for n,v in label.items()}
y=[]
for k in tsukurepo_bow['keyword']:

    y.append(label[k])

# BoW特徴ベクトルをスライシングで取り出す      
X = tsukurepo_bow.iloc[:,4:]

# データセットを訓練データ、テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 訓練データでk nearest neighborの学習を行う
k=10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# テストデータの識別を行う
y_predict=knn.predict(X_test)

# 識別精度を計算
print('predict accuracy :{0} no.of test data :{1}'.format(knn.score(X_test,y_test),len(X_test)))
for p, t in zip(y_predict,y_test):

    print('predict:{0}  true value:{1}'.format(label_inv[p],label_inv[t]) )
```