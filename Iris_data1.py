'''Iris Data Set'''

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the Dataset
dataset = pd.read_csv('/Users/NITISH/Downloads/iris_data1.csv')
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values
#y = pd.get_dummies(y, columns[4], drop_first = False)


'''#Visualising the correlation between different features
plt.figure(figsize=(8,4))
sns.heatmap(dataset.corr(), annot=True, cmap='gist_heat') # draws heatmap with input as correlation matrix calculated by iris.corr() 
plt.show()'''

#Pairplot
sns.set_style('whitegrid')
sns.pairplot(dataset, hue = 'species', size= 2)
plt.show()

#FacetGrid
#sns.FacetGrid(dataset, hue = 'species', size = 3).map(sns.distplot, 'petal_length').add_legend()
#plt.show()

#Label Encoder and One hot encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)
#ohe = OneHotEncoder(categorical_features = [4])
#y = ohe.fit_transform(y).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
x = sd.fit_transform(x)

#Splitting the data into Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

'''#Applying Principal component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variable =pca.explained_variance_ratio_'''  #Accuracy after using PCA - 83.3%

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression as LR
classifier = LR(C = 10.0)
classifier.fit(x_train, y_train)

#Fitting SVC inot the Training set
from sklearn.svm import SVC
clasf = SVC(kernel = 'rbf')
clasf.fit(x_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)
y_clasf = clasf.predict(x_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_pred, y_test)
acc_clasf = accuracy_score(y_clasf, y_test)
