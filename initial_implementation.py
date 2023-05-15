import numpy as np
import sklearn
import csv
import tabulate

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from tabulate import tabulate
from scipy import stats

class AdaBoostClassifier:
    
    def __init__(self,n_estimators):
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
        
rkf = RepeatedKFold(n_repeats=5, n_splits=2, random_state=0)
score = np.empty((10, 5), float)

clf50 = AdaBoostClassifier(n_estimators=50)
clf100 = AdaBoostClassifier(n_estimators=100)
RFclf = RandomForestClassifier(max_depth=2, random_state=0)
GBclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
BGclf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10, random_state=0)
CLFs = { 'AdaBoost n=50': clf50, 'AdaBoost n=100': clf100, 'RandomForest': RFclf, 'GradientBoosting': GBclf, 'Bagging': BGclf }

for i, (train_index, test_index) in enumerate(rkf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    j = 0
    for id, name in enumerate(CLFs):
        clf = CLFs[name]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        score[i][j] = accuracy_score(y_test, prediction)
        j = j + 1

np.save('results.npy', score)
print(tabulate([['Mean score', '{:.1%}'.format(np.mean(score[:,0])), '{:.1%}'.format(np.mean(score[:,1])), '{:.1%}'.format(np.mean(score[:,2])), '{:.1%}'.format(np.mean(score[:,3])), '{:.1%}'.format(np.mean(score[:,4]))], ['Deviation score', '{:.1%}'.format(np.std(score[:,0])), '{:.1%}'.format(np.std(score[:,1])), '{:.1%}'.format(np.std(score[:,2])), '{:.1%}'.format(np.std(score[:,3])), '{:.1%}'.format(np.std(score[:,4]))]], headers=[' ', 'AdaBoost 50xDecisionTrees', 'AdaBoost 100xDecisionTrees', 'Random Forest', 'GradientBoosting', 'Bagging (with DecisionTrees)']))


results = np.load('results.npy')
# print(results)


t = np.zeros((len(CLFs),len(CLFs)))
p = np.zeros((len(CLFs),len(CLFs)))
significant = np.zeros((len(CLFs),len(CLFs))).astype(bool)
advantage = np.zeros((len(CLFs),len(CLFs))).astype(bool)

alpha = 0.05

for i in range(len(CLFs)):
    for j in range(len(CLFs)):
        t[i, j], p[i, j] = stats.ttest_rel(results[:, i], results[:, j])
        if np.mean(results[:, i]) > np.mean(results[:, j]):
            advantage[i, j] = True
        if p[i, j] < alpha: 
            significant[i, j] = True

final = advantage * significant

# print("Statystyka:")
# print(t)
# print("P-wartości:")
# print(p)
# print("Przewaga:")
# print(advantage)
# print("Istotność statycztyczna:")
# print(significant)
# print("Końcowa macierz:\n")
# print(final)

means = np.mean(results, axis=0)

for i in range(len(means)):
    for j in range(len(means)):
        if final[i][j]: print(list(CLFs.keys())[i], "with", round(means[i],3), "better than", list(CLFs.keys())[j], "with", round(means[j],3))



#print('The scikit-learn version is {}.'.format(sklearn.__version__))
