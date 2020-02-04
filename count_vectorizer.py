import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import cross_val_score

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')


df = pd.read_csv("english_literature_base_test.txt.csv", sep = '\t')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['sentence'])

data = [[], []]
data[0] = X.toarray()
data[1] = list(df['intent'])


dt_model = DecisionTreeClassifier()
gaussianNB__model = GaussianNB()
knn_model = KNeighborsClassifier()
svm_model = svm.SVC(gamma='scale')
rf_model = RandomForestClassifier()
#
# # dt_model.fit(data[0], data[1])
# gaussianNB__model.fit(data[0], data[1])
# knn_model.fit(data[0], data[1])
# svm_model.fit(data[0], data[1])
# random_forest = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

int_list = [2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150,200,1000]
dt_param_grid = {'criterion':['gini','entropy'],
                 'splitter':['best', 'random'],
                 'max_depth':int_list,
                 'min_samples_split':int_list,
                 'min_samples_leaf':int_list
                 }

knn_param_grid = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150,200,483],
                  'weights':['uniform', 'distance'],
                  'algorithm':['ball_tree', 'kd_tree']

}

c_param = np.arange(0.2, 0.4, 0.2)#np.arange(0.2, 10.5, 0.2)
svm_param_grid = {'C':c_param,
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                  'degree': [0, 1, 2, 3, 4, 5],
                  'gamma': ['scale', 'auto']
}

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
rf_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
                 }

dt_model_search = GridSearchCV(dt_model, param_grid = dt_param_grid, scoring = 'recall_weighted')
dt_model_search.fit(data[0], data[1])
scores = cross_val_score(dt_model_search, data[0], data[1], scoring = 'recall_weighted')
print(dt_model_search.best_params_)
print("Recall DecisionTreeClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(gaussianNB__model, data[0], data[1], scoring = 'recall_weighted')
print("Recall GaussianNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

knn_model_search = GridSearchCV(knn_model, param_grid = knn_param_grid, scoring = 'recall_weighted')
# try:
knn_model_search.fit(data[0], data[1])
# except ValueError:
#     print("erro de paramentro")
print(knn_model_search.best_params_)
scores = cross_val_score(knn_model_search, data[0], data[1], scoring = 'recall_weighted')
print("Recall KNeighborsClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

svm_model_search = GridSearchCV(svm_model, param_grid = svm_param_grid, scoring = 'recall_weighted')
svm_model_search.fit(data[0], data[1])
scores = cross_val_score(svm_model_search, data[0], data[1], scoring = 'recall_weighted')
print(svm_model_search.best_params_)
print("Recall SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

rf_model_search = GridSearchCV(rf_model, param_grid = rf_param_grid, scoring = 'recall_weighted')
rf_model_search.fit(data[0], data[1])
scores = cross_val_score(rf_model_search, data[0], data[1], scoring = 'recall_weighted')
print(rf_model_search.best_params_)
print("Recall RandomForestClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))