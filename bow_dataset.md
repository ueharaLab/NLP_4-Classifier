# 教師ラベル付きBoW
[tsukurepo_bow_vectorizer.py](tsukurepo_bow_vectorizer.py)を実行すると教師ラベル（杏仁豆腐、シュークリーム、プリン）がついたBoWを作成できる。
- データ数は約500(tsukurepo_df.csv)
- 名詞、動詞、形容詞を抽出[tokenizer.py](tokenizer.py)　動詞は基本形に直している
- 'する', 'なる'などありきたりな語彙（ストップワード）を除外するような処理を加えた。
- 意味のない記号を除外するために日本語文字だけにフィルタリングするなどの改善を加えている（以下コーディング参照）
- これを実行すると約4200次元のBoWになる。

### 演習
countvectorizerのパラメータをmin_df=0.05, max_df=0.5に設定してBoWを作り直し、k nearest neighborを再学習して、識別制度を確認せよ。（BoWは何次元になっているか）
``` python
tsukurepo_df = pd.read_csv('./data/tsukurepo_df.csv', encoding='ms932', sep=',',skiprows=0)
tsukurepo_texts = tsukurepo_df['tsukurepo'].values.tolist()

texts_list=[]
for text in tsukurepo_texts:
    # 日本語は全角に、英数字は半角に揃える処理
    text=unicodedata.normalize('NFKC',text)
    # 日本語文字列だけを取り出す
    text=re.findall('[一-龥ぁ-んァ-ンー々]+',text )     
    text= ''.join(text)
    
    texts_list.append(text)


 
vectorizer = CountVectorizer(tokenizer=tokenize)  # <2>
vec=vectorizer.fit(texts_list)  # <3>
bow = vectorizer.transform(texts_list)  # <4>
print(vec.vocabulary_)
print(bow)
print(bow.toarray())


tsukurepo_bow = pd.DataFrame(bow.toarray(), columns=vectorizer. get_feature_names_out())


