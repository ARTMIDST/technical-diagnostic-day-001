"""\
Title: Iris Dataset Model Comparison Analysis
Author: Afia Konadu
Date: October 2026
Description: Evaluates and compares Logistic Regression and Decision Tree models.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_and_preprocess():
	"""Load Iris data and return a stratified train/test split."""
	iris = load_iris(as_frame=True)
	return train_test_split(
		iris.data,
		iris.target,
		test_size=0.2,
		random_state=42,
		stratify=iris.target,
	)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
	"""Fit a classifier and display its evaluation metrics."""
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	print(f"\n=== {model_name} Evaluation ===")
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, predictions))
	print("\nClassification Report:")
	print(classification_report(y_test, predictions))


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = load_and_preprocess()

	logistic_regression = make_pipeline(
		StandardScaler(), LogisticRegression(max_iter=200, random_state=42)
	)
	evaluate_model(
		logistic_regression,
		X_train,
		X_test,
		y_train,
		y_test,
		"Logistic Regression",
	)

	decision_tree = DecisionTreeClassifier(random_state=42)
	evaluate_model(
		decision_tree,
		X_train,
		X_test,
		y_train,
		y_test,
		"Decision Tree",
	)
