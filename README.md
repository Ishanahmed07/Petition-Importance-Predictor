# Petition Importance Predictor
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

This repository contains the code and documentation for **COMP1804 Task 2**, a machine learning project focused on predicting the importance of petitions ("important" or "not_important") using a semi-supervised approach. The project leverages a Logistic Regression model with TF-IDF vectorization and additional features like petition topics and deviation across regions, achieving high accuracy on a small labeled dataset.

## Project Overview
The goal of this project is to predict the `petition_importance` label for 8,898 petitions from the dataset `comp1804_coursework_dataset_24-25.csv`. Only 100 rows were manually labeled as importance, while the remaining 8,798 are unlabeled. The approach uses a supervised model trained on the labeled data, tuned with GridSearchCV, and then applied to predict importance for the full dataset.

### Key Features
- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, and lemmatization using NLTK.
- **Feature Engineering**: 
  - TF-IDF vectorization of `petition_text`.
  - One-hot encoding of `has_entity`, `petition_topic`, and `petition_status`.
  - Numeric feature: `deviation_across_regions`.
- **Model**: Logistic Regression with class balancing and hyperparameter tuning via GridSearchCV.
- **Evaluation**: Achieves 90% test accuracy on labeled data.

## Dataset
The dataset (`comp1804_coursework_dataset_24-25-copy.csv`) is downloaded from Google Drive using `gdown`. It contains 8,898 rows with columns like `petition_text`, `has_entity`, `petition_topic`, `petition_status`, `deviation_across_regions`, and `petition_importance` (mostly NaN). The project predicts `petition_importance` for all rows and exports results to `predictions.csv`.

## Requirements
To run this project, install the required Python packages:
```bash
pip install gdown pandas numpy scikit-learn nltk
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a Pull Request.
