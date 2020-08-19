from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# TODO: we need to generate more features
# TODO: implement it compeletly -> https://scikit-learn.org/stable/modules/tree.html

class kNearestNeighbor(object):
    def __init__(self, fileName="passenger-list.csv"):
        # reading data
        self.df = pd.read_csv(fileName)

    def splitData(self, x_list=['Sex', 'Category', 'Age', 'Country'], y_list=['Survived']):
        print(x_list)
        X = self.df[x_list]
        y = self.df[y_list]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def oneHotData(self, src='Sex'):
        tempDf = pd.get_dummies(self.df[src])
        self.df[tempDf.columns] = tempDf
        return list(tempDf.columns)

    def trainAgent(self):
        self.clf = tree.DecisionTreeClassifier() 
        self.clf.fit(self.X_train, self.y_train)

    def testModel(self):
        self.y_pred = self.clf.predict(self.X_test)
        print("Decision tree model accuracy(in %):", metrics.accuracy_score(self.y_test, self.y_pred)*100)

    def getProb(self, xTest):
        return self.clf.predict_proba(self.xTest)

    def plotConfusionMatrix(self):
        self.y_test = np.array(list(self.y_test['Survived']))
        TT = ((self.y_pred == self.y_test) & (self.y_test == 1)).sum()
        FF = ((self.y_pred == self.y_test) & (self.y_test == 0)).sum()
        TF = ((self.y_pred != self.y_test) & (self.y_test == 1)).sum()
        FT = ((self.y_pred != self.y_test) & (self.y_test == 0)).sum()
        print("actual value")
        print(TT, TF)
        print(FT, FF)

    def encoder(self, src='Country'):
        label_encoder = LabelEncoder()
        self.df[src] = label_encoder.fit_transform(self.df[src])

    def plot(self):
        tree.plot_tree(self.clf)

model = kNearestNeighbor()
model.encoder()
model.encoder('Sex')
model.encoder('Category')
print(model.df)
# xList = ['Age']
# xList += (model.oneHotData())
# xList += (model.oneHotData('Country'))
# xList += (model.oneHotData('Category'))
model.splitData()
model.trainAgent()
model.testModel()
model.plotConfusionMatrix()
model.plot()