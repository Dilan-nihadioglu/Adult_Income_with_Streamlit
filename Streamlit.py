import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
#from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import GridSearchCV


st.title('Adult-Income')
st.markdown("Bu proje, Adult Income veri setini kullanarak bireylerin yıllık gelirinin 50K’nın üzerinde mi yoksa altında mı olduğunu tahmin etmeyi amaçlar.")
df = pd.read_csv("adult.csv")
df = df.replace('?', np.nan)
df = df.dropna()

#-------------------------------------------------
if st.checkbox('Show dataframe'):
    st.write(df)
#-------------------------------------------------
desc = df.describe().T
if st.checkbox('Show describe statistic'):
    st.write(desc)
#--------------------------------------------------
# Streamlit app
st.title('Dataset Visualization')

# Get list of categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

# Sidebar: Categorical data visualization
st.sidebar.header('Categorical Data Visualization')
categorical_column = st.sidebar.selectbox('Select categorical column', categorical_columns, index=0 if categorical_columns else 0)
chart_type_categorical = st.sidebar.selectbox('Select chart type for categorical data', ['Bar Chart', 'Pie Chart'])

# Sidebar: Numerical data visualization
st.sidebar.header('Numerical Data Visualization')
numerical_column = st.sidebar.selectbox('Select numerical column', numerical_columns, index=0 if numerical_columns else 0)
chart_type_numerical = st.sidebar.selectbox('Select chart type for numerical data', ['Histogram', 'Box Plot'])

# Generate visualizations for categorical data
if categorical_columns:
    st.subheader(f'{chart_type_categorical} for {categorical_column}')
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type_categorical == 'Bar Chart':
        sns.countplot(data=df, x=categorical_column, ax=ax)
    elif chart_type_categorical == 'Pie Chart':
        df[categorical_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# Generate visualizations for numerical data
if numerical_columns:
    st.subheader(f'{chart_type_numerical} for {numerical_column}')
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type_numerical == 'Histogram':
        sns.histplot(df[numerical_column], bins=20, ax=ax)
    elif chart_type_numerical == 'Box Plot':
        sns.boxplot(x=df[numerical_column], ax=ax)
    st.pyplot(fig)

# Split the data
X = df.drop('income', axis=1)
y = df['income']

# Identify categorical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

# Encode categorical features with LabelEncoder
label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
# Encode the target variable 'income'
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split yaparken bu etiketli değişkeni kullanın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize models
lgb_model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=100, use_label_encoder=False, eval_metric='logloss')

# Train models and measure time
start_time = time.time()
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start_time

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

# Make predictions and compute probabilities
lgb_pred = lgb_model.predict(X_test)
lgb_probabilities = lgb_model.predict_proba(X_test)[:, 1]

rf_pred = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

xgb_pred = xgb_model.predict(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]

# Compute metrics for all models
lgb_accuracy = accuracy_score(y_test, lgb_pred)
lgb_loss = log_loss(y_test, lgb_probabilities)
lgb_roc_auc = roc_auc_score(y_test, lgb_probabilities)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_loss = log_loss(y_test, rf_probabilities)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_loss = log_loss(y_test, xgb_probabilities)
xgb_roc_auc = roc_auc_score(y_test, xgb_probabilities)

# Streamlit app for Model Performance
st.title('Model Performance Comparison')

# Display metrics for all models in a table
st.subheader('Model Metrics')
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Log Loss', 'ROC AUC', 'Time (seconds)'],
    'LightGBM': [lgb_accuracy, lgb_loss, lgb_roc_auc, lgb_time],
    'XGBoost': [xgb_accuracy, xgb_loss, xgb_roc_auc, xgb_time],
    'Random Forest': [rf_accuracy, rf_loss, rf_roc_auc, rf_time]
    
})
st.table(metrics_df)

# Display classification reports as a table
st.subheader('Classification Reports')
lgb_report_df = pd.DataFrame(classification_report(y_test, lgb_pred, output_dict=True)).transpose()
xgb_report_df = pd.DataFrame(classification_report(y_test, xgb_pred, output_dict=True)).transpose()
rf_report_df = pd.DataFrame(classification_report(y_test, rf_pred, output_dict=True)).transpose()


st.write('LightGBM Classification Report')
st.dataframe(lgb_report_df)

st.write('Random Forest Classification Report')
st.dataframe(rf_report_df)

st.write('XGBoost Classification Report')
st.dataframe(xgb_report_df)

# Dropdown menu for selecting model to display confusion matrix
model_choice = st.selectbox("Select Model to Display Confusion Matrix", ["LightGBM", "Random Forest", "XGBoost"])

# Function to plot confusion matrix
def plot_confusion_matrix(model_name, y_true, y_pred):
    st.subheader(f'{model_name} Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# Display selected model's confusion matrix
if model_choice == "LightGBM":
    plot_confusion_matrix('LightGBM', y_test, lgb_pred)
elif model_choice == "Random Forest":
    plot_confusion_matrix('Random Forest', y_test, rf_pred)
elif model_choice == "XGBoost":
    plot_confusion_matrix('XGBoost', y_test, xgb_pred)
