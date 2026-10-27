"""Reproducible UCI SMS Spam Collection comparative experiment."""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'SMSSpamCollection'
OUT = ROOT / 'results'
SEED = 42

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def make_models(max_features=3000):
    tfidf = lambda: TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    return {
        'Decision Tree': Pipeline([('tfidf', tfidf()), ('classifier', DecisionTreeClassifier(random_state=SEED, max_depth=30))]),
        'KNN': Pipeline([('tfidf', tfidf()), ('classifier', KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute'))]),
        'SVM': Pipeline([('tfidf', tfidf()), ('classifier', LinearSVC(random_state=SEED, max_iter=5000))]),
        'Naive Bayes': Pipeline([('tfidf', tfidf()), ('classifier', MultinomialNB(alpha=1.0))]),
    }

def evaluate(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction, {
        'Accuracy': accuracy_score(y_test, prediction),
        'Precision': precision_score(y_test, prediction, pos_label='spam', zero_division=0),
        'Recall': recall_score(y_test, prediction, pos_label='spam', zero_division=0),
        'F1': f1_score(y_test, prediction, pos_label='spam', zero_division=0),
    }

def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f'Dataset missing: {DATA_PATH}\nExtract the official UCI ZIP and copy SMSSpamCollection to the data folder.')
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'text'], encoding='latin-1', on_bad_lines='error')
    assert set(df['label'].unique()) == {'ham', 'spam'}, 'Unexpected labels in dataset'
    print('Original messages:', len(df))
    df = df.dropna(subset=['text', 'label']).drop_duplicates(subset=['text']).copy()
    df['text_clean'] = df['text'].map(clean_text)
    df = df[df['text_clean'].ne('')]
    print('After removing duplicate texts / empty rows:', len(df))
    print('Class counts:\n', df.label.value_counts().to_string())
    x_train, x_test, y_train, y_test = train_test_split(df['text_clean'], df['label'], test_size=0.2,
                                                       random_state=SEED, stratify=df['label'])
    print(f'\nTraining={len(x_train)}, testing={len(x_test)}')
    rows = []
    predictions = pd.DataFrame({'message': df.loc[x_test.index, 'text'].values, 'actual': y_test.values})
    for name, model in make_models().items():
        pred, metrics = evaluate(model, x_train, x_test, y_train, y_test)
        rows.append({'Model': name, **metrics})
        predictions[name] = pred
        cm = confusion_matrix(y_test, pred, labels=['ham', 'spam'])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam']).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name + ' Confusion Matrix')
        fig.tight_layout()
        fig.savefig(OUT / f'confusion_{name.lower().replace(" ", "_")}.png', dpi=160)
        plt.close(fig)
        print('\n' + name + '\n' + classification_report(y_test, pred, digits=4, zero_division=0))
        tn, fp, fn, tp = cm.ravel()
        print(f'Confusion counts: TN={tn} FP={fp} FN={fn} TP={tp}')
    results = pd.DataFrame(rows).sort_values('F1', ascending=False)
    results.to_csv(OUT / 'model_comparison.csv', index=False)
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    print('\nCOMPARISON (spam is the positive class):\n', results.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nBest by spam F1:', results.iloc[0]['Model'])
    ax = results.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(kind='bar', figsize=(10,5), ylim=(0,1.05), rot=0)
    ax.set_ylabel('Score (0–1)')
    ax.set_title('Same holdout test set: model comparison')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT / 'model_comparison.png', dpi=160)
    plt.close()
    # One controlled what-if: change ONLY TF-IDF max_features; retain identical data split and SVM model.
    whatif = []
    for count in (500, 3000, 5000):
        model = make_models(max_features=count)['SVM']
        pred, metrics = evaluate(model, x_train, x_test, y_train, y_test)
        whatif.append({'max_features': count, **metrics})
    whatif_df = pd.DataFrame(whatif)
    whatif_df.to_csv(OUT / 'what_if_tfidf_features.csv', index=False)
    print('\nWHAT-IF: SVM max_features\n', whatif_df.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    plt.figure(figsize=(7,4))
    for metric in ('Precision','Recall','F1'):
        plt.plot(whatif_df['max_features'], whatif_df[metric], marker='o', label=metric)
    plt.ylim(0,1.05)
    plt.xlabel('TF-IDF max_features')
    plt.ylabel('Spam-class score')
    plt.title('What-if: TF-IDF vocabulary size with SVM')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'what_if_experiment.png', dpi=160)
    plt.close()
    print('\nSaved plots and CSV results in', OUT)

if __name__ == '__main__':
    main()
