# Streamlit App for Training Model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress the specific Streamlit cache deprecation warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='st.cache is deprecated')

# Suppress warnings globally
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Custom CSS for footer and image styling
st.markdown(
    """
    <style>
    .main {
        margin-left: 10%;  /* Adjust left margin */
        margin-right: 10%; /* Adjust right margin */
    }
    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 1000;
    }
    .footer img {
        height: 30px;
        margin-right: 10px;
    }
    .footer a {
        color: #6c757d;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    /* Image styling */
    img {
        max-width: 50%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the `main` class to the Streamlit container
st.markdown('<div class="main">', unsafe_allow_html=True)

# Footer HTML
st.markdown(
    """
    <div class="footer">
        <span>
            <img src="https://teknik.warmadewa.ac.id/storage/uploads/teknik-komputer-logo.jpg" alt="Warmadewa Computer Engineering">
            <a href="https://www.teknik.warmadewa.ac.id/teknik-komputer">© Warmadewa Computer Engineering Team</a>.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('../dataset/cardiovascular-study-dataset-predict-heart-disease/train.csv')

df = load_data()
st.title("Heart Disease Prediction Training")
st.write("Explore feature selection and train a machine learning model to predict heart disease.")

# Data Preprocessing
st.subheader("Data Preprocessing")

# Missing Value Handling
df_copy = df.copy()
features_to_fill = ['glucose', 'education', 'BPMeds', 'totChol', 'cigsPerDay', 'BMI', 'heartRate']
methods = ['median', 'mode', 'mode', 'median', 'median', 'median', 'median']

for feature, method in zip(features_to_fill, methods):
    if method == 'median':
        df_copy[feature] = df_copy[feature].fillna(df_copy[feature].median())
    elif method == 'mode':
        df_copy[feature] = df_copy[feature].fillna(df_copy[feature].mode()[0])

# Encoding Categorical Variables
le = LabelEncoder()
df_copy['sex'] = le.fit_transform(df_copy['sex'])
df_copy['is_smoking'] = le.fit_transform(df_copy['is_smoking'])

# Feature Selection
features = [
    'age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]
default_features = ['age', 'education', 'sex', 'is_smoking']
target = 'TenYearCHD'

selected_features = st.multiselect("Select Features for Training", features, default=default_features)

if not selected_features:
    st.error("Please select at least one feature.")
else:
    X = df_copy[selected_features]
    y = df_copy[target]

    # SMOTE Oversampling
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

    # 1. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        correlation = df_copy.corr()
        plt.figure(figsize=(12, 8))  # Adjust size
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title("Feature Correlation Matrix")
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
        with col2:  # Center the plot
            st.pyplot(plt)

    # 2. Feature Distribution    
    st.subheader("Feature Distribution")
    if st.checkbox("Show Feature Distributions"):
        # Create 3 columns to show the plots
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column widths        
        # Loop over the selected features and display their distributions in columns
        for i, feature in enumerate(selected_features):
            plt.figure(figsize=(10, 6))  # Create a new figure for each plot
            sns.histplot(df_copy[feature], kde=True, color="blue", bins=30)
            plt.title(f"Distribution of {feature}")

            # Display each plot in the respective column
            if i % 3 == 0:
                with col1:
                    st.pyplot(plt)
            elif i % 3 == 1:
                with col2:
                    st.pyplot(plt)
            else:
                with col3:
                    st.pyplot(plt)

    # 3. Feature vs Target Analysis
    st.subheader("Feature vs Target Analysis")
    if st.checkbox("Show Feature vs Target Analysis"):
        col1, col2, col3 = st.columns([1, 1, 1])
        for i, feature in enumerate(selected_features):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df_copy['TenYearCHD'], y=df_copy[feature], palette="Set2")
            plt.title(f"{feature} vs TenYearCHD")
            plt.xlabel("Ten Year CHD (0: No, 1: Yes)")
            plt.ylabel(feature)
            # Display each plot in the respective column
            if i % 3 == 0:
                with col1:
                    st.pyplot(plt)
            elif i % 3 == 1:
                with col2:
                    st.pyplot(plt)
            else:
                with col3:
                    st.pyplot(plt)

    # Training Button
    if st.button("Train Model"):
        # Model Training
        st.subheader("Model Training")
        classifier = RandomForestClassifier(random_state=42)
        grid_values = {'n_estimators': [50, 65, 80, 95, 120], 'max_depth': [3, 5, 7, 9, 12]}
        GSclassifier = GridSearchCV(classifier, param_grid=grid_values, scoring='roc_auc', cv=5)

        # Fit the model
        st.write("Training the model...")
        GSclassifier.fit(X_train, y_train)
        best_params = GSclassifier.best_params_
        # st.write(f"Best Parameters: {best_params}")

        # Final Model Training
        classifier = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        classifier.fit(X_train, y_train)

        # Evaluation
        st.subheader("Model Evaluation")
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)        
        y_test_proba = classifier.predict_proba(X_test)[:, 1]

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Test Accuracy: {test_accuracy:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
        with col2:  # Center the plot
            st.pyplot(plt)

        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_test_proba):.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
        with col2:  # Center the plot
            st.pyplot(plt)

        # Feature Importance
        st.subheader("Feature Importance")
        importances = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.write(feature_importance_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title("Feature Importance")
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
        with col2:  # Center the plot
            st.pyplot(plt)

        # 6. Model Performance Comparison
        st.subheader("Model Performance Comparison")
        feature_combinations = [
            selected_features[:i] for i in range(1, len(selected_features) + 1)
        ]
        performances = []

        for combination in feature_combinations:
            X_temp = X[combination]
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_temp_sm, y_sm = smote.fit_resample(X_temp, y)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X_temp_sm, y_sm, test_size=0.2, random_state=42
            )
            classifier.fit(X_train_temp, y_train_temp)
            y_test_pred_temp = classifier.predict(X_test_temp)
            acc = accuracy_score(y_test_temp, y_test_pred_temp)
            performances.append((combination, acc))

        performance_df = pd.DataFrame(performances, columns=["Feature Combination", "Accuracy"])
        st.write(performance_df)

# Close the div
st.markdown('</div>', unsafe_allow_html=True)