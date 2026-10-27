# Comparative Study of Text Classification Algorithms for SMS Spam Detection

**Course Assignment Report**  
**Domain**: Artificial Intelligence and Machine Learning / Natural Language Processing  
**Dataset**: UCI Machine Learning Repository SMS Spam Collection  
**Models Evaluated**: Multinomial Naïve Bayes, Decision Tree, K-Nearest Neighbours (KNN)  

---

## 1. Abstract
Short Message Service (SMS) spam poses ongoing security, privacy, and productivity risks for mobile phone subscribers. This project presents an empirical comparative study evaluating three distinct machine learning algorithms—Multinomial Naïve Bayes, Decision Tree, and K-Nearest Neighbours (KNN)—for automated SMS spam classification. Using the benchmark UCI SMS Spam Collection dataset, all messages undergo text cleaning, deduplication, and Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction using word unigrams and bigrams. Classifiers are trained and assessed strictly under an 80:20 stratified train-test holdout split (`random_state=42`) to prevent data leakage. 

Our experimental findings reveal that **K-Nearest Neighbours (KNN)** achieves the highest spam-class F1-score (**0.8963**) and overall accuracy (**97.58%**), followed closely by **Multinomial Naïve Bayes** (F1: **0.8851**, Accuracy: **97.39%**) which uniquely attained **100% precision** (zero false positives). The **Decision Tree** classifier achieved an F1-score of **0.8494** with an accuracy of **96.23%**, suffering from 18 false positives due to rigid single-word decision boundaries. A controlled what-if experiment on KNN reveals that increasing TF-IDF vocabulary features from 500 to 5,000 boosts spam F1-score from 0.8745 to 0.9098, confirming the benefit of fine-grained lexical representations.

---

## 2. Introduction
SMS communication remains a universally accessible, low-latency communication channel. However, its low cost and high open rates make it a prime target for unsolicited promotional messages, financial fraud, phishing scams, and malware distribution. Unlike standard email spam, SMS text messages present unique Natural Language Processing (NLP) challenges:
- Constrained message length (typically ≤ 160 characters).
- Informal grammar, phonetic slang, contractions, and intentional obfuscation (e.g., "win", "w!n", "c@ll").
- Heavy class imbalance, where legitimate personal messages ("ham") significantly outnumber malicious spam.

This investigation explores how statistical, probabilistic, distance-based, and tree-based classification paradigms behave when categorizing sparse SMS textual representations.

---

## 3. Problem Statement
Given an unlabeled incoming SMS text message $M$, construct a robust machine learning classification function:
$$f(M) \in \{\text{ham}, \text{spam}\}$$
that accurately detects fraudulent or unsolicited spam messages while strictly minimizing **False Positives** (misclassifying a genuine, critical message as spam) and **False Negatives** (allowing malicious spam into the user inbox).

---

## 4. Objectives
1. **End-to-End Pipeline**: Develop an automated text processing and classification pipeline using Python and scikit-learn.
2. **Leakage-Free Feature Engineering**: Implement TF-IDF vectorization inside a pipeline where vocabulary extraction and inverse document frequencies are fitted solely on training data.
3. **Multi-Model Comparison**: Empirically evaluate and compare:
   - Multinomial Naïve Bayes
   - Decision Tree
   - K-Nearest Neighbours (KNN)
4. **Rigorous Evaluation**: Benchmark all models on the identical holdout test set using Accuracy, Precision, Recall, F1-score (positive class = spam), and Confusion Matrices.
5. **Hyperparameter Sensitivity (What-If)**: Analyze how varying the vocabulary capacity (`max_features` = 500, 3000, 5000) impacts KNN classification trade-offs.

---

## 5. Dataset Description
The experiment utilizes the official **UCI SMS Spam Collection Dataset** (Almeida, Gómez Hidalgo, and Yamakami, 2012):
- **Source**: UCI Machine Learning Repository (`data/SMSSpamCollection`).
- **Format**: Tab-separated text file without headers (`label \t text`).
- **Raw Volume**: 5,572 raw records.
- **Deduplication**: 406 duplicate entries and empty lines were removed to prevent evaluation bias.
- **Cleaned Corpus**: **5,166 unique messages**.
  - **Ham (legitimate)**: 4,513 messages (**87.36%**)
  - **Spam**: 653 messages (**12.64%**)
  - **Class Imbalance Ratio**: Approximately 6.91 : 1.

```
Class Distribution:
ham     4513  (87.36%)
spam     653  (12.64%)
Total   5166  (100.0%)
```

---

## 6. Data Preprocessing
Raw SMS texts contain inconsistent capitalization, phone numbers, hyperlinks, and punctuation. The text preprocessing function `clean_text(text)` standardizes each message through:
1. **Case Normalization**: Conversion to lowercase (`text.lower()`).
2. **URL Tokenization**: Replacing web links (`http://...`, `www...`) with a unified token `' URL '`.
3. **Digit Tokenization**: Replacing sequences of numbers (e.g., claim numbers, shortcodes, phone numbers) with a unified token `' NUM '`.
4. **Punctuation & Special Character Removal**: Stripping non-alphabetic symbols (`[^a-z\s]`), retaining only clean alphabetical words and space tokens.
5. **Whitespace Normalization**: Collapsing multiple consecutive spaces into single whitespace and trimming boundaries.

**Stratified Train-Test Split**:
To preserve the 87.36% : 12.64% class distribution across splits, scikit-learn's `train_test_split` is applied with `stratify=df['label']`, `test_size=0.20`, and `random_state=42`:
- **Training Subset (80%)**: 4,132 messages (3,610 ham, 522 spam).
- **Holdout Testing Subset (20%)**: 1,034 messages (903 ham, 131 spam).

---

## 7. TF-IDF Feature Extraction
Text strings are mapped to numeric feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Configuration details:
- **N-gram Range**: `(1, 2)` (extracts single words and adjacent word pairs to preserve short phrase context, e.g., "free entry", "urgent call").
- **Minimum Document Frequency (`min_df`)**: 2 (filters out singleton typos that occur only once across the entire corpus).
- **Maximum Features (`max_features`)**: 3,000 top vocabulary terms ranked by term frequency across the corpus.
- **Data Leakage Prevention**: Encapsulated within a scikit-learn `Pipeline`. The vectorizer calculates vocabulary and document frequencies strictly on `x_train` and merely transforms `x_test`.

---

## 8. Proposed System Architecture

```
   Raw UCI SMSSpamCollection
               │
               ▼
   Data Cleaning & Deduplication (5,166 unique texts)
               │
               ▼
   Stratified 80:20 Train-Test Split (Seed = 42)
        ├── Training (4,132)
        └── Testing (1,034)
               │
               ▼
   TF-IDF Vectorization Pipeline (ngram=(1,2), max_features=3000)
        ├── Multinomial Naïve Bayes (alpha=1.0)
        ├── Decision Tree (max_depth=30, seed=42)
        └── K-Nearest Neighbours (k=5, metric=cosine, brute)
               │
               ▼
   Evaluation on 1,034 Test Samples (Holdout)
        ├── Accuracy, Precision, Recall, F1-Score
        ├── Confusion Matrix Generation
        └── What-If Experiment (KNN: 500 vs 3000 vs 5000 features)
               │
               ▼
   Output Results (CSVs, Confusion Matrices, Plots)
```

---

## 9. Implementation of the Three Algorithms

### 9.1 Multinomial Naïve Bayes (`MultinomialNB`)
- **Mathematical Principle**: Applies Bayes' theorem with the assumption of conditional feature independence given class $c$:
  $$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{n} P(x_i \mid c)$$
- **Smoothing**: Laplace additive smoothing ($\alpha = 1.0$) prevents zero-probability errors for unseen terms:
  $$P(x_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$
- **Characteristics**: Fast training, probabilistic calibration, and exceptional resistance to false alarms.

### 9.2 Decision Tree Classifier (`DecisionTreeClassifier`)
- **Mathematical Principle**: Builds an axis-aligned recursive binary partition tree. Splits are chosen to maximize Gini Impurity reduction:
  $$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$
- **Hyperparameter Configuration**: `max_depth = 30`, `random_state = 42`.
- **Characteristics**: Highly interpretable if-then rules, non-parametric, but prone to variance and fragile on high-dimensional sparse representations.

### 9.3 K-Nearest Neighbours (`KNeighborsClassifier`)
- **Mathematical Principle**: An instance-based non-parametric classifier. Given test vector $\mathbf{u}$, it computes distance to all stored training instances $\mathbf{v}$ and assigns the majority class among the $k$ nearest neighbors:
  $$\text{Cosine Distance}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Hyperparameter Configuration**: `n_neighbors = 5`, `metric = 'cosine'`, `algorithm = 'brute'`.
- **Characteristics**: Effective angle-based semantic comparison; robust to document length variations due to L2 normalization.

---

## 10. Performance Evaluation Metrics
Because the dataset is imbalanced (87.4% ham vs 12.6% spam), spam detection treats **spam** as the positive class ($P$) and **ham** as the negative class ($N$).

1. **Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Precision (Spam)**: Proportion of detected spams that were truly spam (critical for preventing legitimate message loss):
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
3. **Recall (Spam)**: Proportion of all actual spam messages correctly intercepted:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
4. **F1-Score (Spam)**: Harmonic mean balancing precision and recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 11. Comparative Results Table

The table below presents the **exact empirical results** generated on the holdout test set (1,034 samples):

| Rank | Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **K-Nearest Neighbours (KNN)** | **0.9758** | 0.9818 | 0.8244 | **0.8963** |
| **2** | **Multinomial Naïve Bayes** | 0.9739 | **1.0000** | 0.7939 | 0.8851 |
| **3** | **Decision Tree** | 0.9623 | 0.8594 | **0.8397** | 0.8494 |

*Source file: `results/model_comparison.csv`*

---

## 12. Performance Comparison Graph
The comparative performance bar chart generated by the system is shown below:

```
  1.00 ┤  ██ █  ██       ██ █  ██       ██ █  ██
       │  ██ █  ██       ██ █  ██       ██ █  ██
  0.80 ┤  ██ █  ██       ██ █  ██       ██ █  ██
       │  ██ █  ██       ██ █  ██       ██ █  ██
  0.60 ┤  ██ █  ██       ██ █  ██       ██ █  ██
       │  ██ █  ██       ██ █  ██       ██ █  ██
  0.40 ┤  ██ █  ██       ██ █  ██       ██ █  ██
       │  ██ █  ██       ██ █  ██       ██ █  ██
  0.20 ┤  ██ █  ██       ██ █  ██       ██ █  ██
       └────────────────────────────────────────
            KNN        Naive Bayes    Decision Tree
         [■ Acc  ■ Prec  ■ Recall  ■ F1]
```
*(Rendered graphical plot available at: `results/model_comparison.png`)*

---

## 13. Confusion Matrix Analysis

Each algorithm was evaluated on the identical 1,034 holdout test samples (903 True Ham, 131 True Spam):

```
                       KNN                     Naive Bayes                Decision Tree
               Pred Ham   Pred Spam       Pred Ham   Pred Spam       Pred Ham   Pred Spam
Actual Ham  │    901          2    │  │    903          0    │  │    885         18    │
Actual Spam │     23        108    │  │     27        104    │  │     21        110    │
```

### Detailed Error Analysis:
1. **Multinomial Naïve Bayes (0 False Positives, 27 False Negatives)**:
   - **Zero False Positives**: Naïve Bayes achieved an ideal precision of 1.0000. It never misclassified a genuine personal SMS as spam.
   - **Higher False Negatives**: 27 actual spam messages evaded detection.
   - *Representative Failure*: Text containing conversational greetings ("Hi Jordan, how r u? Im home from abroad and lonely...") contained predominant ham tokens that numerically outweighed the spam probability terms.
2. **K-Nearest Neighbours (2 False Positives, 23 False Negatives)**:
   - Misclassified only 2 ham messages as spam (Precision = 98.18%).
   - *Representative FP*: A legitimate academic message containing a standalone URL (`staff.science.nus.edu.sg/~phyhcmk/teaching/pc1323`) had high cosine similarity with promotional link spam.
3. **Decision Tree (18 False Positives, 21 False Negatives)**:
   - Generated by far the highest false positive count (18 legitimate messages marked spam).
   - *Representative FP*: A message like *"I to am looking forward to all the sex cuddling.. Only two more sleeps"* was flagged as spam because the tree created a split directly on the token "sex".

---

## 14. What-If Experiment: TF-IDF Vocabulary Capacity (`max_features`)

To investigate feature dimensionality impact, **KNN** was tested across vocabulary ceilings of 500, 3,000, and 5,000 features while retaining identical training and testing splits:

| `max_features` | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **500** | 0.9700 | 0.9310 | 0.8244 | 0.8745 |
| **3000** | 0.9758 | 0.9818 | 0.8244 | 0.8963 |
| **5000** | **0.9787** | **0.9823** | **0.8473** | **0.9098** |

*Source file: `results/what_if_tfidf_features.csv`*

### Observations:
- **At 500 features**: The vocabulary is too restricted. Rare spam indicator tokens are discarded, causing false positives to rise and precision to drop to 93.10%.
- **At 5,000 features**: Incorporating specific unigrams and bigrams allows KNN's cosine distance to capture distinct spam phrasings, increasing recall from 82.44% to 84.73% and elevating F1-score to 0.9098.

---

## 15. Results and Discussion

### Why the Algorithms Diverge:
1. **Curse of Dimensionality vs Distance Metric**:
   KNN with Euclidean distance typically degrades in high-dimensional text. However, utilizing **cosine metric** normalizes vector length, converting the problem into angular orientation where document lengths do not skew neighbor determination.
2. **Conditional Independence vs Real Language**:
   Naïve Bayes makes the naive assumption that words are conditionally independent. In reality, spam tokens correlate (e.g., "call" + "now", "claim" + "prize"). This causes Naïve Bayes to underestimate spam posteriors when words are subtle, but makes its high-confidence predictions virtually faultless (100% precision).
3. **Greedy Single-Feature Splitting**:
   Decision Trees make orthogonal splits based on single keywords. Because text classification requires synthesizing multi-word cues, individual words frequently co-occur in both benign and malicious contexts, causing high false alarm rates.

---

## 16. Technical Viva Quick Reference

| Question | Recommended Answer |
| :--- | :--- |
| **Which algorithm was best overall?** | K-Nearest Neighbours (KNN) using cosine metric, achieving the highest F1-score (**0.8963**) and highest accuracy (**97.58%**). |
| **Which algorithm is safest for user experience?** | Multinomial Naïve Bayes. It produced **0 False Positives** (100% precision), guaranteeing that no legitimate SMS was lost. |
| **Why not just rely on Accuracy?** | With an 87.4% ham majority, a baseline model that labels everything as ham achieves 87.4% accuracy but has a 0% spam F1-score. F1-score and Precision are essential. |
| **How did you prevent data leakage?** | TF-IDF vectorization was placed inside a Pipeline; vocabulary and inverse document frequencies were calculated strictly on `x_train` and only applied to `x_test`. |
| **Why does Decision Tree have low precision?** | It splits on single words (e.g., "free", "call"). Genuine messages with those words trigger false alarms (18 FPs in test). |

---

## 17. Conclusion and Future Scope

### Conclusion
This comparative study successfully built, evaluated, and analyzed three machine learning classifiers on the benchmark UCI SMS Spam Collection. KNN (cosine distance) proved to be the most balanced performer (F1: 0.8963), while Multinomial Naïve Bayes emerged as the most conservative spam filter (Precision: 1.0000). Decision Trees proved suboptimal for sparse TF-IDF vectors due to overfitting individual trigger words.

### Future Scope
1. **Ensemble Methods**: Evaluating Random Forests and Gradient Boosted Trees (XGBoost/LightGBM) to reduce single-tree variance.
2. **Deep Learning & Transformers**: Implementing lightweight transformer models (e.g., DistilBERT) to analyze sequential semantics beyond n-gram bags.
3. **Edge Deployment**: Quantizing the Naïve Bayes or KNN model for real-time on-device Android/iOS SMS filtering.

---

## 18. References
1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2012). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng'11.
2. UCI Machine Learning Repository. *SMS Spam Collection Data Set*. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
