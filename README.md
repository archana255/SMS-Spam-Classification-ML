# Comparative Study of Text Classification Algorithms for SMS Spam Detection

A reproducible machine learning project comparing text classification algorithms on the benchmark **UCI SMS Spam Collection** dataset.

---

## 1. Project Overview

- **Algorithms Evaluated**:
  1. **Multinomial Naïve Bayes** (`MultinomialNB`)
  2. **Decision Tree** (`DecisionTreeClassifier`)
  3. **K-Nearest Neighbours** (`KNeighborsClassifier`)
- **Dataset**: UCI SMS Spam Collection (5,166 unique messages after deduplication).
- **Split**: 80:20 stratified holdout split (`random_state=42`).
- **Feature Extraction**: TF-IDF (unigrams & bigrams, `min_df=2`, `max_features=3000`) fitted only on training data to prevent leakage.
- **Detailed Academic Report**: See [REPORT.md](REPORT.md) for full theory, math, confusion matrix breakdowns, and literature references.

---

## 2. Experimental Results Summary

*(Holdout Test Set: 1,034 samples — 903 Ham, 131 Spam. Positive class = **Spam**)*

| Rank | Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **K-Nearest Neighbours (KNN)** | **0.9758** | 0.9818 | 0.8244 | **0.8963** |
| **2** | **Multinomial Naïve Bayes** | 0.9739 | **1.0000** | 0.7939 | 0.8851 |
| **3** | **Decision Tree** | 0.9623 | 0.8594 | **0.8397** | 0.8494 |

### Confusion Matrix Counts (Test Set)
- **KNN**: TN = 901, FP = 2, FN = 23, TP = 108
- **Naïve Bayes**: TN = 903, **FP = 0**, FN = 27, TP = 104 *(Zero false alarms)*
- **Decision Tree**: TN = 885, FP = 18, FN = 21, TP = 110

### What-If Experiment (KNN Vocabulary Capacity)
| `max_features` | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **500** | 0.9700 | 0.9310 | 0.8244 | 0.8745 |
| **3000** | 0.9758 | 0.9818 | 0.8244 | 0.8963 |
| **5000** | **0.9787** | **0.9823** | **0.8473** | **0.9098** |

---

## 3. Directory Structure

```
SMS_Spam_Classification_Project/
├── data/
│   └── SMSSpamCollection          # Official UCI dataset
├── results/
│   ├── model_comparison.csv       # Comparative metrics table
│   ├── model_comparison.png       # 3-model comparative bar chart
│   ├── test_predictions.csv      # Test samples with predictions
│   ├── what_if_tfidf_features.csv # KNN feature experiment table
│   ├── what_if_experiment.png     # KNN feature experiment plot
│   ├── confusion_knn.png          # KNN confusion matrix plot
│   ├── confusion_naive_bayes.png  # Naïve Bayes confusion matrix plot
│   └── confusion_decision_tree.png# Decision Tree confusion matrix plot
├── main.py                        # Executable ML pipeline
├── REPORT.md                      # Academic report (theory, viva notes)
├── requirements.txt               # Pinned library dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules (.venv, cache)
```

---

## 4. Setup & Execution Instructions (Windows PowerShell)

### Prerequisites
- Python 3.11+ installed.

### Step 1: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or directly use `.\.venv\Scripts\python.exe`)*

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Run the Pipeline
```powershell
python main.py
```
This executes data cleaning, training, evaluation, what-if experiment, and regenerates all CSVs and visualization plots inside `results/`.

---

## 5. Software Environment Record

- **Python**: 3.11.9
- **scikit-learn**: 1.9.1
- **pandas**: 2.3.3
- **matplotlib**: 3.11.2
- **numpy**: 2.4.6
- **scipy**: 1.17.1

---

## 6. Viva Cheat Sheet

1. **Why is F1-score preferred over Accuracy?**  
   The dataset has an 87.4% majority of "ham" messages. A baseline model predicting "ham" every time would achieve 87.4% accuracy but a 0% spam F1-score.
2. **What is a False Positive vs. False Negative in spam filtering?**  
   - **False Positive (FP)**: A genuine personal message misclassified as spam (very bad user experience).
   - **False Negative (FN)**: A spam message that slips into the inbox (minor annoyance).
3. **Why did Naïve Bayes achieve 1.0000 Precision?**  
   Laplace smoothing and Bayesian likelihood ratios required strong, undisputed spam token evidence before flagging a text as spam, producing zero false alarms (FP = 0).
4. **Why did KNN win overall F1?**  
   By using **cosine metric**, KNN normalized vector lengths and compared semantic orientations, avoiding the Euclidean distance degradation often seen in sparse high-dimensional text.
