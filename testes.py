from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df_intents = pd.read_csv('intents.csv')
sentences = df_intents["sentence"]

vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names())
print(X)