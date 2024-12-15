import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
# 'exc(% matplotlip in line)'  # Line 5: This line seems to be a comment or error; consider removing it.
import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab  # Line 6: Unused import, can be removed
import seaborn as sns  

# Load the dataset
disease_df = pd.read_csv("C:/Users/EMMA/Downloads/framingham (2).csv")

# Drop the 'education' column as it is not needed for analysis
disease_df.drop(['education'], inplace=True, axis=1)  # Line 10: Removed unnecessary space
# Rename 'male' column to 'Sex_male' for clarity
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)  # Line 11: Removed unnecessary space

# Remove rows with NaN/Null values
disease_df.dropna(axis=0, inplace=True)  # Line 14: Removed unnecessary space
# Display the first few rows and the shape of the DataFrame
print(disease_df.head(), disease_df.shape)
# Count the occurrences of each value in the 'TenYearCHD' column
print(disease_df.TenYearCHD.value_counts())

# Import train_test_split for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split  # Line 18: Moved import statement here

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
print('Accuracy of the model is =', accuracy_score(y_test, y_pred))  # Line 42: Corrected print statement