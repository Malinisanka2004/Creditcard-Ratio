import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os

# Step 1: Create a sample dataset if not present
def create_sample_data(file_name):
    """
    Creates a sample CSV file for testing if it doesn't exist.
    """
    if not os.path.exists(file_name):
        num_rows = 1000
        np.random.seed(42)

        # Synthetic dataset
        data = {
            "Age": np.random.randint(18, 70, size=num_rows),
            "Income": np.random.randint(20000, 120000, size=num_rows),
            "LoanAmount": np.random.randint(1000, 50000, size=num_rows),
            "CreditScore": np.random.randint(300, 850, size=num_rows),
            "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], size=num_rows),
            "Gender": np.random.choice(["Male", "Female"], size=num_rows),
            "Approval": np.random.choice([0, 1], size=num_rows)  # Target column
        }

        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        print(f"Sample dataset '{file_name}' created successfully!")
    else:
        print(f"Dataset '{file_name}' already exists.")

# Step 2: Load the dataset
def load_data(file_name):
    """
    Loads the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_name)
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return None

# Step 3: Data Preprocessing
def preprocess_data(data):
    """
    Handles missing values, encodes categorical data, and scales numerical data.
    """
    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)

    # Encode categorical data
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Split features and target
    X = data.drop('Approval', axis=1)  # Replace 'Approval' with the actual target column
    y = data['Approval']

    # Scale numerical data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, label_encoders

# Step 4: Model Training
def train_models(X_train, y_train):
    """
    Trains multiple models and returns them in a dictionary.
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

# Step 5: Evaluate Models
def evaluate_models(models, X_test, y_test):
    """
    Evaluates models on test data and prints classification reports.
    """
    for name, model in models.items():
        print(f"\n{name} Performance:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Handle Class Imbalance with SMOTE
def handle_imbalance(X, y):
    """
    Handles imbalanced datasets using SMOTE.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Step 7: Analyze Feature Importance
def analyze_feature_importance(models, feature_names):
    """
    Analyzes feature importance for tree-based models.
    """
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)
            print(f"\n{name} Feature Importance:")
            print(importance_df)

            # Plotting feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df)
            plt.title(f"{name} Feature Importance")
            plt.show()

# Step 8: Main Function
def main():
    file_name = "credit_card_data.csv"

    print("Checking and creating dataset if needed...")
    create_sample_data(file_name)

    print("\nLoading data...")
    data = load_data(file_name)
    if data is None:
        return

    print("\nPreprocessing data...")
    X, y, label_encoders = preprocess_data(data)

    print("\nHandling class imbalance using SMOTE...")
    X_resampled, y_resampled = handle_imbalance(X, y)

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print("\nTraining models...")
    models = train_models(X_train, y_train)

    print("\nEvaluating models...")
    evaluate_models(models, X_test, y_test)

    print("\nAnalyzing feature importance...")
    analyze_feature_importance(models, data.drop('Approval', axis=1).columns)

if _name_ == "_main_":
    main()
