# Logistic Regression Model for Heart Disease Prediction

This project applies a Logistic Regression model to predict the likelihood of a patient developing Coronary Heart Disease (CHD) within ten years, using the **Framingham Heart Study** dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Setup and Requirements](#setup-and-requirements)
- [Key Features](#key-features)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Overview
Heart disease is one of the leading causes of death globally. Predicting its occurrence can help in early diagnosis and treatment. In this project, we preprocess the dataset, build a Logistic Regression model, and evaluate its performance using various metrics.

---

## Dataset
The dataset used is the **Framingham Heart Study dataset**, which contains features related to demographic, lifestyle, and medical factors that contribute to heart disease risk.

- **Source:** Framingham Heart Study
- **Key Features:**
  - `age`: Age of the patient
  - `Sex_male`: Gender (1 = male, 0 = female)
  - `cigsPerDay`: Number of cigarettes smoked per day
  - `totChol`: Total cholesterol level
  - `sysBP`: Systolic blood pressure
  - `glucose`: Glucose level
  - `TenYearCHD`: Target variable (1 = CHD likely, 0 = CHD unlikely)

---

## Project Workflow
1. **Data Preprocessing**:
   - Drop unnecessary columns (e.g., `education`).
   - Handle missing values by removing rows with null entries.
   - Normalize the feature set using `StandardScaler`.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of the target variable using a count plot.

3. **Model Development**:
   - Split the dataset into training (70%) and testing (30%) sets.
   - Train a Logistic Regression model.

4. **Model Evaluation**:
   - Assess model performance using accuracy score, confusion matrix, and classification report.
   - Visualize the confusion matrix using a heatmap.

---

## Setup and Requirements
### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `scipy`

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python heart_disease_prediction.py
   ```

---

## Key Features
- **Data Preprocessing**:
  - Handles missing data.
  - Renames columns for clarity.
- **Logistic Regression Model**:
  - Predicts the likelihood of developing heart disease.
- **Visualization**:
  - Displays a count plot of CHD occurrences.
  - Generates a heatmap for the confusion matrix.
- **Performance Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

## Results
### Model Accuracy
The accuracy of the Logistic Regression model is:
```
Accuracy of the model is = [Your Accuracy Score]
```

### Confusion Matrix
![Confusion Matrix](confusion_matrix_plot.png)

### Classification Report
The detailed classification report includes precision, recall, F1-score, and support.

---

## Acknowledgements
- Dataset: [Framingham Heart Study](https://biolincc.nhlbi.nih.gov/studies/framcohort/)
- Visualization inspired by Seaborn's documentation.

---

*This project was developed and modified by Emmanuel.*
