# SMS Spam Detection — Comparative ML Study

Dataset: UCI SMS Spam Collection, https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Windows VS Code setup
1. Install Python 3.11+ and VS Code with Microsoft's Python extension.
2. Download ZIP: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
3. Extract `SMSSpamCollection` (no file extension) and put it in the `data` directory beside `main.py`.
4. Open this folder in VS Code.
5. In VS Code > Terminal > New Terminal (PowerShell):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python main.py
   ```
   If PowerShell blocks activation, run without activating:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe main.py
   ```
6. Inspect `results/model_comparison.csv`, `results/what_if_tfidf_features.csv`, confusion matrix PNGs and charts. Results are generated from YOUR execution, not supplied as invented values.
7. Commit source + README + your generated results to GitHub (and submit to Google Classroom). Do not commit the `.venv` folder.

## Protocol
- Remove exact duplicate SMS texts before splitting; lowercase / replace URLs, digits, punctuation.
- Stratified 80/20 holdout split, random_state=42.
- TF-IDF unigrams and bigrams, min_df=2, max_features=3000, learned only on training data (via Pipeline).
- Classifiers evaluated:
  1. Multinomial Naïve Bayes (`MultinomialNB(alpha=1.0)`)
  2. Decision Tree (`DecisionTreeClassifier(max_depth=30, random_state=42)`)
  3. K-Nearest Neighbours (`KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute')`)
- Accuracy, precision, recall, and F1 based on the **spam** (positive) class.
- Best performer selected using spam F1; investigate precision/recall tradeoffs.
- What-if: change only max_features (500, 3000, 5000) for KNN on the same split.

## Notes for the viva
- False positive: a legitimate ham message marked spam. False negative: a spam message marked ham.
- Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = harmonic mean of precision and recall.
- Class imbalance means accuracy alone can be misleading. Naïve Bayes leverages conditional word probabilities and often yields very high precision; Decision Trees create interpretable rule splits but can struggle with high-dimensional sparsity; KNN computes cosine similarity against stored training instances but is computationally intensive at inference time.
