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

le=LabelEncoder()
df_copy['sex']=le.fit_transform(df_copy['sex'])
df_copy['is_smoking']=le.fit_transform(df_copy['is_smoking'])

df_copy.head()
# df_copy.drop(['id'],axis=1,inplace=True) #Id is not useful for Model training.

fig = plt.figure(figsize = (20,15))
ax = fig.gca()
df_copy.hist(ax = ax)

# plt.figure(figsize=(15,8))
# correlation = df_copy.corr()
# sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Select only numeric columns
numeric_df = df_copy.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# Filter only numeric columns
numeric_columns = df_copy.select_dtypes(include=[float, int]).columns[:-1]

for i in numeric_columns:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df_copy.boxplot(column=i, by='TenYearCHD', ax=ax)
    ax.set_ylabel(i)
plt.show()

