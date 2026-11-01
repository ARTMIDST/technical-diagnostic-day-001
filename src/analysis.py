"""\
Title: Iris Dataset Model Comparison Analysis
Author: Afia Konadu
Date: October 2026
Description: Evaluates and compares Logistic Regression and Decision Tree models.
"""

from sklearn.datasets import load_iris  # type: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingModuleSource]
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]
from sklearn.pipeline import make_pipeline  # type: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingModuleSource]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[reportMissingModuleSource]

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 200


def load_dataset():
	"""Load the Iris features and target."""
	iris = load_iris(as_frame=True)
	return iris.data, iris.target


def split_dataset(features, target):
	"""Return a reproducible, stratified train/test split."""
	return train_test_split(
		features,
		target,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=target,
	)


def load_and_preprocess():
	"""Load Iris data and return a stratified train/test split."""
	return split_dataset(*load_dataset())


def build_models():
	"""Create the classifiers compared in this analysis."""
	return {
		"Logistic Regression": make_pipeline(
			StandardScaler(),
			LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE),
		),
		"Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
	}


def evaluate_model(model, X_train, X_test, y_train, y_test):
	"""Fit a classifier and return its evaluation metrics."""
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	return confusion_matrix(y_test, predictions), classification_report(
		y_test, predictions
	)


def display_evaluation(model_name, metrics):
	"""Display a model's confusion matrix and classification report."""
	matrix, report = metrics
	print(f"\n=== {model_name} Evaluation ===")
	print("Confusion Matrix:")
	print(matrix)
	print("\nClassification Report:")
	print(report)


def run_analysis():
	"""Train and evaluate each classifier on the shared train/test split."""
	X_train, X_test, y_train, y_test = load_and_preprocess()

	for model_name, model in build_models().items():
		metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
		display_evaluation(model_name, metrics)


if __name__ == "__main__":
	run_analysis()
