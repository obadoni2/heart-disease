import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
# 'exc(% matplotlip in line)'  # This line seems to be a comment or error; consider removing it.
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab  # Unused import, can be removed
import seaborn as sns  

# Load the dataset
disease_df = pd.read_csv("C:/Users/EMMA/Downloads/framingham (2).csv")

# Drop the 'education' column as it is not needed for analysis
disease_df.drop(['education'], inplace=True, axis=1)  # Removed unnecessary space

# Rename 'male' column to 'Sex_male' for clarity
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)  # Removed unnecessary space

# Remove rows with NaN/Null values
disease_df.dropna(axis=0, inplace=True)  # Removed unnecessary space

# Display the first few rows and the shape of the DataFrame
print(disease_df.head(), disease_df.shape)

# Count the occurrences of each value in the 'TenYearCHD' column
print(disease_df.TenYearCHD.value_counts())

# Import train_test_split for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split  # Moved import statement here

# Prepare the feature set (X) and target variable (y)
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the feature set
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.3, random_state=4)

# Print the shapes of the training and testing sets
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Visualize the count of patients affected with CHD
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df,
             palette="BuGn_r")
plt.show()

# Import accuracy_score for model evaluation
from sklearn.metrics import accuracy_score  # Ensure this is imported

# Fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Print the accuracy of the model
print('Accuracy of the model is =', accuracy_score(y_test, y_pred))  # Corrected print statement

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