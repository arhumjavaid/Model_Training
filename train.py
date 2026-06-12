
# train.py — Optimized Pipeline


import re
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


# CONFIG

LABELS = ['admiration', 'anger', 'disgust', 'fear', 'hope',
          'joy', 'love', 'pride', 'sadness']

RANDOM_STATE = 42
TEST_SIZE = 0.2

# STEP 1: Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)         # strip # keep word
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# STEP 2: Load & Prepare Data

print("📦 Loading dataset...")
df = pd.read_csv("dataset.csv")
assert all(col in df.columns for col in ['Tweets (text)', 'Emotions (Multi-labeled)']), \
    "❌ Column name mismatch — check dataset.csv headers!"

texts = df['Tweets (text)'].apply(preprocess).tolist()

# Parse multi-labels
def parse_labels(label_str):
    return [l.strip() for l in str(label_str).split(',')]

raw_labels = df['Emotions (Multi-labeled)'].apply(parse_labels).tolist()

# Binarize labels (must match LABELS order)
mlb = MultiLabelBinarizer(classes=LABELS)
Y = mlb.fit_transform(raw_labels)

print(f"✅ Dataset: {len(texts)} samples, {Y.shape[1]} labels")
print(f"   Label order: {mlb.classes_.tolist()}")


# STEP 3: Train/Val Split

X_train_raw, X_val_raw, Y_train, Y_val = train_test_split(
    texts, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"   Train: {len(X_train_raw)}, Val: {len(X_val_raw)}")


# STEP 4: Vectorization

print("\n🔤 Building FeatureUnion (word + char TF-IDF)...")
vectorizer = FeatureUnion([
    ('word', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
        analyzer='word'
    )),
    ('char', TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=10000,
        sublinear_tf=True,
        min_df=3,
        analyzer='char_wb'
    ))
])
X_train = vectorizer.fit_transform(X_train_raw)
X_val   = vectorizer.transform(X_val_raw)
print(f"   Feature matrix: {X_train.shape}")


# STEP 5: Train Classifier

print("\n🤖 Training Ensemble: LinearSVC + LogisticRegression...")
# Model 1: LinearSVC (calibrated for probabilities)
svc_clf = CalibratedClassifierCV(
    LinearSVC(C=0.5, max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
    cv=3
)
model_svc = OneVsRestClassifier(svc_clf, n_jobs=-1)
model_svc.fit(X_train, Y_train)

# Model 2: LogisticRegression
model_lr = OneVsRestClassifier(
    LogisticRegression(C=5, max_iter=1000, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),
    n_jobs=-1
)
model_lr.fit(X_train, Y_train)
print("   Training complete ✅")


# STEP 6: Evaluate (flat 0.5 threshold)

print("\n📊 Evaluating on validation set...")

# Ensemble: average probabilities from both models
probs_svc = model_svc.predict_proba(X_val)
probs_lr  = model_lr.predict_proba(X_val)
if isinstance(probs_svc, list):
    probs_svc = np.array([p[:, 1] for p in probs_svc]).T
if isinstance(probs_lr, list):
    probs_lr  = np.array([p[:, 1] for p in probs_lr]).T
probs = (probs_svc + probs_lr) / 2.0   # shape (n_val, 9)

Y_pred_flat = (probs > 0.5).astype(int)

f1_micro = f1_score(Y_val, Y_pred_flat, average='micro', zero_division=0)
f1_macro = f1_score(Y_val, Y_pred_flat, average='macro', zero_division=0)
f1_per_label = f1_score(Y_val, Y_pred_flat, average=None, zero_division=0)

print(f"   F1 Micro : {f1_micro:.4f}")
print(f"   F1 Macro : {f1_macro:.4f}")
print("   Per-label F1:")
for label, score in zip(LABELS, f1_per_label):
    print(f"     {label:<12}: {score:.4f}")


# PHASE 2: Per-label Threshold Tuning

print("\n🎯 Tuning per-label thresholds on val set...")
thresholds = []
for i, label in enumerate(LABELS):
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 17):
        preds_t = (probs[:, i] > t).astype(int)
        f1_t = f1_score(Y_val[:, i], preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
    thresholds.append(round(best_thresh, 2))
    print(f"     {label:<12}: threshold={best_thresh:.2f}  F1={best_f1:.4f}")

thresholds = np.array(thresholds)

# Evaluate with tuned thresholds
Y_pred_tuned = (probs > thresholds).astype(int)
f1_micro_tuned = f1_score(Y_val, Y_pred_tuned, average='micro', zero_division=0)
f1_macro_tuned = f1_score(Y_val, Y_pred_tuned, average='macro', zero_division=0)

print(f"\n   [Tuned] F1 Micro : {f1_micro_tuned:.4f}  (was {f1_micro:.4f})")
print(f"   [Tuned] F1 Macro : {f1_macro_tuned:.4f}  (was {f1_macro:.4f})")


# REFIT ON FULL DATA (maximize performance)

print("\n🔁 Re-fitting BOTH models on FULL dataset...")
X_full = vectorizer.fit_transform(texts)
model_svc.fit(X_full, Y)
model_lr.fit(X_full, Y)
print("   Refit complete ✅")


# STEP 7: Save model.pkl

print("\n💾 Saving model.pkl...")
bundle = {
    "vectorizer": vectorizer,
    "classifier": {"svc": model_svc, "lr": model_lr},
    "thresholds": thresholds
}
joblib.dump(bundle, "model.pkl", compress=3)
print("   Saved: model.pkl ✅")
print(f"   Keys: {list(bundle.keys())}")


# STEP 8: Quick sanity check

print("\n🔍 Sanity check — loading model.pkl and running predict()...")
from model_wrapper import MyModel
m = MyModel()
sample = [
    "I'm so proud of what we achieved together, this feels amazing!",
    "This situation is frustrating and honestly makes me really upset.",
    "Even though things are uncertain, I still believe everything will work out."
]
out = m.predict(sample)
print(f"   Output shape: {out.shape}  (expected: ({len(sample)}, 9))")
print(f"   Output dtype: {out.dtype}")
for text, row in zip(sample, out):
    predicted = [LABELS[i] for i, v in enumerate(row) if v == 1]
    print(f"   → \"{text[:50]}...\" → {predicted}")

print("\n🏁 DONE — model is ready for submission!")
