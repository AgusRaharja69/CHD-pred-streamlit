# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
# Import sklearn and IPython tools
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../dataset/cardiovascular-study-dataset-predict-heart-disease/train.csv')
df.head()
# Copy original data
df_copy = df.copy()

# Missing Value Count Function
def show_missing():
    missing = df_copy.columns[df_copy.isnull().any()].tolist()
    return missing

# Fill Missing Data
features_to_fill = ['glucose', 'education', 'BPMeds', 'totChol', 'cigsPerDay', 'BMI', 'heartRate']
methods = ['median', 'mode', 'mode', 'median', 'median', 'median', 'median']

for feature, method in zip(features_to_fill, methods):
    print(f'{feature.capitalize()} Missing Data Filling...')
    if method == 'median':
        df_copy[feature] = df_copy[feature].fillna(df[feature].median())
    elif method == 'mode':
        df_copy[feature] = df_copy[feature].fillna(df[feature].mode()[0])

# Training Preparation
le=LabelEncoder()
df_copy['sex']=le.fit_transform(df_copy['sex'])
df_copy['is_smoking']=le.fit_transform(df_copy['is_smoking'])
df_copy.head()
# df_copy.drop(['id'],axis=1,inplace=True) #Id is not useful for Model training.
fig = plt.figure(figsize = (20,15))
ax = fig.gca()
df_copy.hist(ax = ax)

# Select only numeric columns
numeric_df = df_copy.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation = numeric_df.corr()

# Filter only numeric columns
numeric_columns = df_copy.select_dtypes(include=[float, int]).columns[:-1]

X=df_copy[['age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose']].copy()
y=df_copy['TenYearCHD'].copy()

X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.2, random_state = 0)

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
# X_sm, y_sm = smote.fit(X,y)
y_sm=pd.DataFrame(y_sm)
y_sm.value_counts()

X_train, X_test, y_train, y_test = train_test_split( X_sm,y_sm , test_size = 0.2, random_state = 0)

'''
Random Forest
'''
classifier = RandomForestClassifier() # For GBM, use GradientBoostingClassifier()
grid_values = {'n_estimators':[50, 65, 80, 95,120], 'max_depth':[3, 5, 7,9,12]}
GSclassifier = GridSearchCV(classifier, param_grid = grid_values, scoring = 'roc_auc', cv=5)

# Fit the object to train dataset
GSclassifier.fit(X_train, y_train)

bestvalues=GSclassifier.best_params_
GSclassifier.best_params_

classifier = RandomForestClassifier(max_depth=bestvalues['max_depth'],n_estimators=bestvalues['n_estimators']) # For GBM, use GradientBoostingClassifier()
classifier.fit(X_train, y_train)

y_train_preds_rf =  classifier.predict(X_train)
y_test_preds_rf= classifier.predict(X_test)

# Obtain accuracy on train set
accuracy_score(y_train,y_train_preds_rf)
# Obtain accuracy on test set
accuracy_score(y_test,y_test_preds_rf)

# Get the confusion matrix for both train and test
plt.figure(figsize=(5,3))
labels = ['Negative', 'Positive']
print('Confusion Matrix for training Data')
cm = confusion_matrix(y_train, y_train_preds_rf)
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

# Get the confusion matrix for both train and test
plt.figure(figsize=(5,3))
labels = ['Retained', 'Churned']
print('Confusion Matrix for Test Data')
cm = confusion_matrix(y_test, y_test_preds_rf)
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

y_rf_predict_pro=classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_rf_predict_pro)

roc_auc_score(y_test,y_rf_predict_pro)

plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC curve')
plt.show()

features = X.columns
importances = classifier.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,9))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()