from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time


print("hello")

df = pd.read_csv('preco_etanol_2004_2019.csv', sep='\t')
data = []
pesos = []
print(df.shape)

start = time.time()
for i in df.index:
    if df.iloc[i]['preco'] >= 50:
        df.at[i, 'preco'] = int(df.at[i, 'preco']) / 1000
        df.to_csv('preco_etanol_2004_2019.csv.csv', index=False)
    if i%100 == 0:
        end = time.time()
        print(i, end - start)

# sentences = df_intents["sentence"]
#
# vectorizer = TfidfVectorizer(analyzer='word')
# X = vectorizer.fit_transform(sentences)
# # print(vectorizer.get_feature_names())
# # print(X)

