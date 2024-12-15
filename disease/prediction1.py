import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
# 'exc(% matplotlip in line)'  # This line seems to be a comment or error; consider removing it.
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab  # Unused import, can be removed
import seaborn as sns  

# dataset 
disease_df = pd.read_csv("C:/Users/EMMA/Downloads/framingham (2).csv")
disease_df.drop(['education'], inplace=True, axis=1)  # Removed unnecessary space
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)  # Removed unnecessary space

#removing NaN/Null values
disease_df.dropna(axis=0, inplace=True)  # Removed unnecessary space
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# Train-and-Test -Split
from sklearn.model_selection import train_test_split  # Moved import statement here

#splitting the dataset into Test and Train Sets
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# normalization of the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# counting no. of patients affected with CHD
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df,
             palette="BuGn_r")
plt.show()

laste = disease_df['TenYearCHD'].plot()
plt.show(laste)


from sklearn.linear_model  import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict (X_test)

# Evalution and accuracy
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', accuracy_score,(y_test, y_pred))


from sklearn. metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['predicted:0', 'Predicted:1'],
                            index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot = True, fmt='d' , cmap= "Greens")

plt.show()
print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred))

# this code is modified by emmanuel 