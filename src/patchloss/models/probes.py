"""Linear and non-linear classical probes with train-only fitting."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np


def probe(train_x, train_y, test_x, test_y, nonlinear=False, seed=0):
    tr = np.asarray(train_x).reshape(len(train_x), -1)
    te = np.asarray(test_x).reshape(len(test_x), -1)
    clf = (
        MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=100, random_state=seed, early_stopping=True, n_iter_no_change=8
        )
        if nonlinear
        else LogisticRegression(max_iter=200, random_state=seed)
    )
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(tr, train_y)
    pred = pipe.predict(te)
    return {
        "accuracy": float(np.mean(pred == test_y)),
        "error": float(np.mean(pred != test_y)),
        "classifier": "mlp" if nonlinear else "logistic",
    }
