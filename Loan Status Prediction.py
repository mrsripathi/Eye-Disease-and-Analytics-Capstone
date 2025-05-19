import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load training dataset
df_train = pd.read_csv("Loan_Status_train.csv")

# Fill missing values in training data
df_train["LoanAmount"].fillna(df_train["LoanAmount"].median(), inplace=True)
df_train["Loan_Amount_Term"].fillna(df_train["Loan_Amount_Term"].mode()[0], inplace=True)
df_train["Credit_History"].fillna(df_train["Credit_History"].mode()[0], inplace=True)
df_train["Self_Employed"].fillna(df_train["Self_Employed"].mode()[0], inplace=True)
df_train["Dependents"].fillna(df_train["Dependents"].mode()[0], inplace=True)
df_train["Gender"].fillna(df_train["Gender"].mode()[0], inplace=True)
df_train["Married"].fillna(df_train["Married"].mode()[0], inplace=True)

# Convert categorical features
df_train = pd.get_dummies(df_train, columns=["Gender", "Property_Area", "Self_Employed", "Married"], dtype=int)
df_train["Loan_Status"] = df_train["Loan_Status"].map({'Y': 1, 'N': 0})
df_train["Education"] = df_train["Education"].map({'Graduate': 1, 'Not Graduate': 0})
df_train["Dependents"] = df_train["Dependents"].replace('3+', '3').astype(int)

# Feature Scaling
scaler = StandardScaler()
df_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(
    df_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
)

# 📊 Visualizations
sns.countplot(x='Loan_Status', data=df_train)
plt.title("Loan Status Distribution")
plt.show()

sns.boxplot(x='LoanAmount', y='ApplicantIncome', data=df_train)
plt.title("Loan Amount vs Applicant Income")
plt.show()

# 🚀 Prepare Features and Target
X = df_train.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df_train['Loan_Status']

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X, y)
lr_pred = lr_model.predict(X)
print("Logistic Regression Accuracy (Train):", accuracy_score(y, lr_pred))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)
dt_pred = dt_model.predict(X)
print("Decision Tree Accuracy (Train):", accuracy_score(y, dt_pred))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X, y)
rf_pred = rf_model.predict(X)
print("Random Forest Accuracy (Train):", accuracy_score(y, rf_pred))

# XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)
xgb_pred = xgb_model.predict(X)
print("XGBoost Accuracy (Train):", accuracy_score(y, xgb_pred))
