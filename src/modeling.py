from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state, 
        class_weight='balanced', 
        oob_score=True
    )
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree classifier"""
    model = DecisionTreeClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier"""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    return y_pred, cm

def plot_confusion_matrices(cm1, cm2, cm3, model_names=["Random Forest", "Decision Tree", "Logistic Regression"]):
    """Plot confusion matrices for multiple models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot confusion matrix for model1
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{model_names[0]} Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # Plot confusion matrix for model2
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title(f"{model_names[1]} Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    # Plot confusion matrix for model3
    sns.heatmap(cm3, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title(f"{model_names[2]} Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()

def adjust_threshold(model, X_val, y_val, threshold=0.4):
    """Adjust prediction threshold and evaluate"""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_threshold = (y_pred_proba > threshold).astype(int)
    print(classification_report(y_val, y_pred_threshold))
    cm = confusion_matrix(y_val, y_pred_threshold)
    print(cm)
    
    # Plot confusion matrix
    plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix with Adjusted Threshold')
    plt.show()
    
    return y_pred_threshold, cm
