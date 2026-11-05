# Iris Flower Classification

## Project Overview

This project uses machine learning to classify iris flowers into three species based on their physical measurements. It demonstrates a basic end-to-end classification workflow, from data exploration and preprocessing to model training, evaluation, and comparison.

## Problem Statement

The objective is to identify the species of an iris flower using four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The project treats this as a supervised multiclass classification problem.

## Dataset

The project uses the Iris dataset, a well-known dataset containing measurements from three iris species:

- Setosa
- Versicolor
- Virginica

The dataset contains 150 observations, with four numerical features and one target variable representing the flower species.

The dataset is loaded using Scikit-learn.

## Models Used

Two classification algorithms were implemented and compared.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an observation belonging to each class.

A StandardScaler was included in its processing pipeline to standardize the input features before model training.

### 2. Decision Tree Classifier

A Decision Tree is a supervised learning algorithm that makes predictions using a sequence of decision rules based on the input features.

Unlike Logistic Regression, the Decision Tree model was trained without feature scaling.

## Methodology

The project followed these steps:

1. **Data loading:** Loaded the Iris dataset into a Pandas DataFrame.
2. **Data exploration:** Examined the dataset structure, descriptive statistics, missing values, species distribution, and feature relationships.
3. **Feature and target selection:** Used the four flower measurements as input features and species as the target variable.
4. **Data splitting:** Divided the dataset into 80% training data and 20% testing data using a stratified split and `random_state=42`.
5. **Model training:** Trained Logistic Regression and Decision Tree classifiers using the training dataset.
6. **Model evaluation:** Evaluated both models using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
7. **Model comparison:** Compared their evaluation results to understand their classification performance.

## Model Evaluation and Results

Both models achieved **93% accuracy** on the test dataset.

| Evaluation | Logistic Regression | Decision Tree |
|---|---:|---:|
| Test accuracy | 93% | 93% |

Both models correctly classified 28 out of the 30 test observations.

Their confusion matrices also showed similar classification errors, with one Versicolor and one Virginica observation misclassified by each model.

Although both models achieved the same accuracy, this does not establish that they will perform equally well on every dataset. Their classification reports and confusion matrices provide additional information about their performance.

## Technical Questions and Answers

### Why do we split a dataset into training and testing sets?

The training set teaches the model patterns in the data. A separate test set estimates how well it performs on unseen examples, helping us assess whether it generalizes rather than memorizes.

### What is the difference between classification and regression?

Classification predicts a category, such as spam or not spam. Regression predicts a continuous numeric value, such as temperature or house price.

### Why might a Decision Tree overfit?

A tree can grow very deep and learn rules that fit individual training examples, including noise. This can lead to excellent training results but poor performance on new data. Limiting its depth or pruning it can help.

### What does a confusion matrix tell us that accuracy alone doesn't?

A confusion matrix shows the counts of correct and incorrect predictions for each class. It reveals which classes are confused and the types of errors that a single accuracy score can hide.

### If your model achieves 98% accuracy, does that automatically mean it is a good model?

No. If 98% of examples belong to one class, a model that always predicts that class can reach 98% accuracy while missing every example of the other class. We should also consider class balance, error costs, other relevant metrics, and performance on representative unseen data.

### What is the difference between a model's parameters and hyperparameters?

Parameters are learned from the training data, such as weights in a linear model. Hyperparameters are settings chosen to guide training, such as a tree's maximum depth.

### Why is reproducibility important in machine learning?

Reproducibility makes it possible to repeat experiments, verify findings, and compare approaches reliably. Tracking data and code versions, settings, and random seeds also makes experiments easier to debug.

## Project Structure

```text
technical-diagnostic-day-001/
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── src/
│   └── analysis.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run the Project

### Prerequisites

- Python 3
- pip

### 1. Clone the repository

```bash
git clone https://github.com/ARTMIDST/technical-diagnostic-day-001.git
```

### 2. Navigate into the project directory

```bash
cd technical-diagnostic-day-001
```

### 3. Create and activate a virtual environment (recommended)

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the analysis script

```bash
python src/analysis.py
```

The script loads the Iris dataset, prepares the data, trains both classification models, and displays their evaluation results.

### 6. Explore the notebook

Open the notebook in Jupyter:

```bash
jupyter notebook notebooks/iris_classification.ipynb
```

## Conclusion

This project demonstrates a fundamental machine-learning classification workflow using Python and Scikit-learn.

By implementing and comparing Logistic Regression and Decision Tree classifiers, I gained practical experience in data exploration, preprocessing, model training, evaluation, and Python code organization.

The experiment also demonstrates why model evaluation should go beyond accuracy by considering classification reports and confusion matrices.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Git and GitHub
