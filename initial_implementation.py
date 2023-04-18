import numpy as np
import sklearn
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

class AdaBoostClassifier:
    
    def __init__(self,n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.model_errors = []
        self.alphas = []
        
    def fit(self,X,y):
        
        N = len(y)
        #poczatkowe wagi rowne, przyjete kazda 1/N
        w = np.array([1/N for i in range(N)])
        
        self.classes = np.unique(y)
        k = len(self.classes)
        
        for m in range(self.n_estimators):
            
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            m_y_pred = model.predict(X)
            incorrect = m_y_pred != y
            
            #oblicz pojednczy blad dla modelu
            model_err = np.average(incorrect,weights=w,axis=0)
            
            self.model_errors.append(model_err)
            
            #oblicz alfe (zaufanie)
            alpha = (np.log((1-model_err)/model_err)+np.log(k-1))

            self.alphas.append(alpha)
            
            #zaktualizuj wagi
            w *= np.exp(alpha*incorrect)
            
            self.models.append(model)
            

    def predict(self,X):
        
        y = 0
        
        for m in range(self.n_estimators):
            alpha = self.alphas[m]
            model = self.models[m]  
            y += alpha*model.predict_proba(X)
        
        return self.classes[np.argmax(y,axis=1)]
    
X = np.empty((0, 7), int)
y = np.empty((0, 1), int)
    
with open(r"C:\Users\Admin\Downloads\wifi_localization_csv.txt") as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        X = np.append(X, np.array([[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])]]), axis=0)
        y = np.append(y, int(row[7]))

#print(X)
#print(y)


rkf = RepeatedKFold(n_repeats=5, n_splits=2, random_state=1)
score = []

for i, (train_index, test_index) in enumerate(rkf.split(X, y)):
    clf = AdaBoostClassifier()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score.append(accuracy_score(y_test, prediction))

print('Mean score: {:.1%}'.format(np.mean(score)))

print('Deviation score: {:.1%}'.format(np.std(score)))
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
