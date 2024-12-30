# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import sklearn and IPython tools
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('../dataset/cardiovascular-study-dataset-predict-heart-disease/train.csv')
df.head()

# Data Info
print(df.info())
print(df.describe(include='all').T)

# Copy original data
df_copy = df.copy()

# Missing Value Count Function
def show_missing():
    missing = df_copy.columns[df_copy.isnull().any()].tolist()
    return missing

# Missing Data Counts and Percentages
print('Missing Data Count:')
print(df_copy[show_missing()].isnull().sum().sort_values(ascending=False))
print('--'*50)
print('Missing Data Percentage:')
print(round(df_copy[show_missing()].isnull().sum().sort_values(
    ascending=False) / len(df_copy) * 100, 2))
round(df_copy[show_missing()].isnull().sum().sort_values(
    ascending=False) / len(df_copy) * 100, 2).plot(kind='bar', color=['red', 'yellow', 'blue', 'orange'])
plt.title('Missing Data Percentage by Feature')
plt.show()  # Display the plot

# Fill Missing Data
features_to_fill = ['glucose', 'education', 'BPMeds', 'totChol', 'cigsPerDay', 'BMI', 'heartRate']
methods = ['median', 'mode', 'mode', 'median', 'median', 'median', 'median']

for feature, method in zip(features_to_fill, methods):
    print(f'{feature.capitalize()} Missing Data Filling...')
    if method == 'median':
        df_copy[feature] = df_copy[feature].fillna(df[feature].median())
    elif method == 'mode':
        df_copy[feature] = df_copy[feature].fillna(df[feature].mode()[0])
    print(f'{feature.capitalize()} Missing After Filling:')
    print(df_copy[[feature]].isnull().sum())
    print('--'*50)

# Age Distribution Plot
fig, ax = plt.subplots(figsize=(15, 6))
age_dis = pd.DataFrame(df.groupby(['age'])['id'].count())
sns.barplot(x=age_dis.index, y=age_dis['id'])
plt.ylabel('Counts')
plt.title('Age Distribution')
plt.show()  # Display the plot


plt.rcParams['figure.figsize'] = (15, 5)
df.groupby(['age','TenYearCHD'])['id'].count().unstack().plot(kind='bar')
plt.title('Age wise Effected People')
plt.show()

plt.rcParams['figure.figsize'] = (20, 5)
df.groupby(['age', 'sex'])['TenYearCHD'].count().unstack().plot(kind='bar')
plt.title('Age and Gender wise TenYearCHD Count')
plt.show()  # Ensures this plot is displayed
