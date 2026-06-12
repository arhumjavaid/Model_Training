
import joblib
import numpy as np

# ⚠️ DO NOT CHANGE THIS ORDER
LABELS = ['admiration','anger','disgust','fear','hope',
          'joy','love','pride','sadness']


class MyModel:
    def __init__(self):
        '''
        ⚠️ DO NOT MODIFY THIS METHOD

        Rules:
        - Class name MUST remain 'MyModel'
        - Filename MUST be "model.pkl"
        - Keys inside model.pkl must include:
            "vectorizer" and "classifier"
        '''

        bundle = joblib.load("model.pkl")  # ⚠️ DO NOT CHANGE FILENAME

        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]   # dict: {"svc": ..., "lr": ...} or single model

        # ✅ Optional: load thresholds if present (added in Phase 2)
        self.thresholds = bundle.get("thresholds", None)


    def predict(self, texts):
        '''
        ✅ YOU CAN MODIFY THIS FUNCTION

        Input:
            texts → list of strings

        Output:
            numpy array of shape (n_samples, 9)

        ⚠️ RULES:
        - Must return numpy array
        - Shape MUST be (n_samples, 9)
        - Column order MUST match LABELS
        '''


        # 🔹 STEP 1: Preprocessing
        texts = [str(t).lower() for t in texts]


        # 🔹 STEP 2: Vectorization

        X = self.vectorizer.transform(texts)


        # 🔹 STEP 3: Prediction (probabilities)

        # predict_proba — handle ensemble (dict) or single model
        clf = self.classifier
        if isinstance(clf, dict):
            p1 = clf["svc"].predict_proba(X)
            p2 = clf["lr"].predict_proba(X)
            if isinstance(p1, list):
                p1 = np.array([p[:, 1] for p in p1]).T
            if isinstance(p2, list):
                p2 = np.array([p[:, 1] for p in p2]).T
            probs = (p1 + p2) / 2.0
        else:
            probs = clf.predict_proba(X)
            if isinstance(probs, list):
                probs = np.array([p[:, 1] for p in probs]).T
        # probs shape: (n_samples, 9)


        # 🔹 STEP 4: Apply thresholds

        if self.thresholds is not None:
            preds = (probs > self.thresholds).astype(int)
        else:
            preds = (probs > 0.5).astype(int)

        # Safety: guarantee correct shape
        preds = np.asarray(preds)
        if preds.ndim == 1 or preds.shape[1] != 9:
            preds = preds.reshape(-1, 9)

        return preds