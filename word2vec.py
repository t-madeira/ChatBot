import pandas as pd
import bot_functions
import spacy
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import dot
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score


# nlp = spacy.load('pt_core_news_sm')
nlp = spacy.load('en')
df = pd.read_csv("intents_test.csv", sep = '\t')

sentences = df['sentence']

cleanned_sentence = []

for sentence in sentences:
    sentence = bot_functions.clean_sentence(sentence, nlp)
    cleanned_sentence.append(sentence)
sentences = cleanned_sentence

vocab = []
for sentence in sentences:
    for word in sentence:
        if word not in vocab and word != ' ':
            vocab.append(word)

model = Word2Vec(sentences, min_count=1,size= 50,workers=3, window =3, sg = 1)

print(sentences)
print(model)


# def display_closestwords_tsnescatterplot(model, word, size):
#     arr = np.empty((0, size), dtype='f')
#     word_labels = [word]
#
#
#     close_words = model.similar_by_word(word)
#     arr = np.append(arr, np.array([model[word]]), axis=0)
#     for wrd_score in close_words:
#         wrd_vector = model[wrd_score[0]]
#         word_labels.append(wrd_score[0])
#         arr = np.append(arr, np.array([wrd_vector]), axis=0)
#
#     tsne = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(arr)
#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     plt.scatter(x_coords, y_coords)
#     for label, x, y in zip(word_labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
#     plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
#     plt.show()
#
# display_closestwords_tsnescatterplot(model, 'dia', 50)

sentences_vsm = []
for sentence in sentences:
    sentence_vsm = [0]*50
    for word in sentence:
        sentence_vsm += model[word]
    sentences_vsm.append(sentence_vsm)

data = [[], []]
data[0] = sentences_vsm
data[1] = list(df['intent'])

# 6. Treina o modelo
dt_model = DecisionTreeClassifier()
gaussianNB__model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=15)
svm_model = svm.SVC(gamma='scale')

dt_model.fit(data[0], data[1])
gaussianNB__model.fit(data[0], data[1])
knn_model.fit(data[0], data[1])
svm_model.fit(data[0], data[1])
random_forest = RandomForestClassifier()

scores = cross_val_score(dt_model, data[0], data[1], cv=5)
print("Accuracy DecisionTreeClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(gaussianNB__model, data[0], data[1], cv=5)
print("Accuracy GaussianNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(knn_model, data[0], data[1], cv=5)
print("Accuracy KNeighborsClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(svm_model, data[0], data[1], cv=5)
print("Accuracy SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(random_forest, data[0], data[1], cv=5)
print("Accuracy Random Forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
