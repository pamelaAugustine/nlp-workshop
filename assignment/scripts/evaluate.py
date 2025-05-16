'''
Model evaluation utilities for NLP tasks.
'''
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from train_model import train_model, X, y, newsgroups

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and display confusion matrix."""
    y_pred = model.predict(X_test)
    
    # Generate classification report and save as CSV
    report_dict = classification_report(y_test, y_pred, target_names=newsgroups.target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report.csv", index=True)
    print("Classification report saved as classification_report.csv")
    
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=newsgroups.target_names))
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=newsgroups.target_names, 
                yticklabels=newsgroups.target_names, ax=ax)
    
    plt.xlabel("Predicted Category", labelpad=20)
    plt.ylabel("Actual Category", labelpad=20)
    plt.title("Confusion Matrix - Text Classification", pad=30)
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train model if not already trained
    model, vectorizer, X_test, y_test = train_model(X, y)

    # Evaluate the trained model
    evaluate_model(model, X_test, y_test)
