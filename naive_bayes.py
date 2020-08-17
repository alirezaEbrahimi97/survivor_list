from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#FIXME: naive bayes needs float/integer data, data shall be one hotted 
#Age, AgePart, Sex-Age and Country-Age reduce accuracy

class naiveBayes(object):
    def __init__(self, fileName="passenger-list.csv"):
        #reading data
        self.df = pd.read_csv(fileName)

    def trainAgent(self):
        self.gnb = GaussianNB() 
        self.gnb.fit(self.X_train, self.y_train)

    def splitData(self, x_list=['Country', 'Sex', 'Category', 'AgePart', 'Country-Age'], y_list=['Survived']):
        X = self.df[x_list]
        y = self.df[y_list]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def testModel(self):
        self.y_pred = self.gnb.predict(self.X_test)
        print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(self.y_test, self.y_pred)*100)

    def encoder(self, src='Country'):
        label_encoder = LabelEncoder()
        self.df[src] = label_encoder.fit_transform(self.df[src])

    def generateData(self):
        encodeList = ['Country', 'Sex', 'Category']
        for el in encodeList:
            model.encoder(el)
        
        #brings down the accurecy
        self.df['AgePart'] = self.df['Age'] // 10

        #reduces the accuracy
        self.df['Country-Age'] = self.df['Country'] * 100 + self.df['Age']
        self.df['Sex-Age'] = self.df['Sex'] * 2 + self.df['Age']

    def plotConfusionMatrix(self):
        # print(self.y_pred)
        # print(self.y_test.values)
        # self.y_test = self.y_test.values
        # # equal = self.y_pred == self.y_test
        # TT = ((self.y_pred == self.y_test) & (self.y_test == 1)).sum()
        # FF = ((self.y_pred == self.y_test) & (self.y_test == 0)).sum()
        # TF = ((self.y_pred != self.y_test) & (self.y_test == 1)).sum()
        # FT = ((self.y_pred != self.y_test) & (self.y_test == 0)).sum()
        # print("actual value")
        # print(TT, TF)
        # print(FT, FF)
        print(f"number of true ones: {(self.y_test == 1).sum()}")
        print(f"number of true zeros: {(self.y_test == 0).sum()}")
        print(f"number of predicted ones: {(self.y_pred == 1).sum()}")
        print(f"number of predicted zeros: {(self.y_pred == 0).sum()}")
        # TT = ((self.y_pred == self.y_test) & (self.y_test == 1)).sum() / (self.y_test == 1).sum()
        # FF = ((self.y_pred == self.y_test) & (self.y_test == 0)).sum() / (self.y_test == 0).sum()
        # TF = ((self.y_pred != self.y_test) & (self.y_test == 1)).sum() / (self.y_test == 1).sum()
        # FT = ((self.y_pred != self.y_test) & (self.y_test == 0)).sum() / (self.y_test == 0).sum()
        # print("precentage")
        # print(TT, TF)
        # print(FT, FF)
        # plt.show()
    
model = naiveBayes()

model.generateData()
print(model.df)
model.splitData()
model.trainAgent()
model.testModel()
model.plotConfusionMatrix()
# unq, counts = np.unique(model.df['Country'], return_counts=True)
# print(unq)
#print(counts)