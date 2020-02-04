import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import cross_val_score

df = pd.read_csv("intents_test.csv", sep = '\t')
print (df.columns)
vectorizer = HashingVectorizer()
X = vectorizer.fit_transform(df['sentence'])

data = [[], []]
data[0] = X.toarray()
data[1] = list(df['intent'])


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