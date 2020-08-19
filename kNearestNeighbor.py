from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class kNearestNeighbor(object):
    def __init__(self, fileName="passenger-list.csv"):
        # reading data
        self.df = pd.read_csv(fileName)

    def splitData(self, x_list=['Sex', 'Category', 'F', 'M'], y_list=['Survived']):
        print(x_list)
        X = self.df[x_list]
        y = self.df[y_list]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def oneHotData(self, src='Sex'):
        tempDf = pd.get_dummies(self.df[src])
        self.df[tempDf.columns] = tempDf
        return list(tempDf.columns)

    def trainAgent(self, nNeighbors=3):
        self.knn = KNeighborsClassifier(n_neighbors=nNeighbors) 
        self.knn.fit(self.X_train, self.y_train)

    def testModel(self):
        self.y_pred = self.knn.predict(self.X_test)
        print("K nearest neighbor model accuracy(in %):", metrics.accuracy_score(self.y_test, self.y_pred)*100)

    def getProb(self, xTest):
        return self.knn.predict_proba(self.xTest)

    def plotConfusionMatrix(self):
        self.y_test = np.array(list(self.y_test['Survived']))
        TT = ((self.y_pred == self.y_test) & (self.y_test == 1)).sum()
        FF = ((self.y_pred == self.y_test) & (self.y_test == 0)).sum()
        TF = ((self.y_pred != self.y_test) & (self.y_test == 1)).sum()
        FT = ((self.y_pred != self.y_test) & (self.y_test == 0)).sum()
        print("actual value")
        print(TT, TF)
        print(FT, FF)

model = kNearestNeighbor()
xList = ['Age']
xList += (model.oneHotData())
xList += (model.oneHotData('Country'))
xList += (model.oneHotData('Category'))
model.splitData(x_list=xList)
model.trainAgent()
model.testModel()
model.plotConfusionMatrix()