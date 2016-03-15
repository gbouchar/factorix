
# import numpy as np
from sklearn import metrics
# from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
# from sklearn.svm import l1_min_c
# import tensorflow as tf

# from naga.shared.tf_addons import tf_eval
from factorix.learn_to_update import EmbeddingUpdater, multitask_to_tuples
# from factorix.evaluation import train_test_split


def eval_auc(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def test_logistic_regression_equivalence_learning(verbose=False):
    iris = datasets.load_iris()
    x_mat = iris.data
    y = iris.target
    x_mat = x_mat[y != 2, 0:1]
    y = y[y != 2]
    x_mat -= np.mean(x_mat, 0)
    n_ents = x_mat.shape[1] + 3
    idx_show = [0, 99]
    c = 1e10  # no regularization
    clf1 = linear_model.LogisticRegression(C=c, penalty='l2', tol=1e-6)
    clf1.fit(x_mat, y)
    scores = x_mat.dot(clf1.coef_.T) + clf1.intercept_
    clf2 = EmbeddingUpdater(rank=1, n_ents=n_ents, n_slots=2, reg=1e-10, max_epochs=200, verbose=verbose,
                            preprocessing=multitask_to_tuples)
    clf2.logistic2embeddings(coefs=clf1.coef_, intercept=clf1.intercept_)
    pred, y, nll = clf2.predict(x_mat, y)
    clf2.fit(x_mat, y)
    pred2, y2, nll2 = clf2.predict(x_mat, y)
    clf3 = EmbeddingUpdater(rank=1, n_ents=n_ents, n_slots=2, reg=1e-10, max_epochs=200, verbose=verbose,
                            preprocessing=multitask_to_tuples)
    clf3.fit(x_mat, y)
    pred3, y3, nll3 = clf3.predict(x_mat, y)
    if verbose:
        print('logistic regression prediction:\n', 1.0 / (np.exp(-scores[idx_show]) + 1.0))
        print('EmbeddingUpdater predictions before learning:\n', 1.0 / (np.exp(-pred[idx_show]) + 1.0))
        print('EmbeddingUpdater predictions after learning (oracle initialization): \n',
              1.0 / (np.exp(-pred2[idx_show]) + 1.0))
        print('EmbeddingUpdater predictions after learning (random initialization): \n',
              1.0 / (np.exp(-pred3[idx_show]) + 1.0))
    assert(np.linalg.norm(scores-pred) < 1e-2)
    assert(np.linalg.norm(scores-pred2) < 1e-2)
    assert(np.linalg.norm(scores-pred3) < 1e-2)


def test_logistic_regression_equivalence_prediction(verbose=False):
    iris = datasets.load_iris()
    x_mat = iris.data
    y = iris.target
    x_mat = x_mat[y != 2]
    y = y[y != 2]
    x_mat -= np.mean(x_mat, 0)
    n_ents = x_mat.shape[1] + 3  # number of entities is the dimension with 2 slots and 1 intercept
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    clf1.fit(x_mat, y)
    prob = clf1.predict_proba(x_mat)
    n_show = 3
    scores = x_mat.dot(clf1.coef_.T) + clf1.intercept_
    clf2 = EmbeddingUpdater(rank=1, n_ents=n_ents, n_slots=2, reg=1.0, max_epochs=500, verbose=False,
                            preprocessing=multitask_to_tuples)
    clf2.logistic2embeddings(coefs=clf1.coef_, intercept=clf1.intercept_)
    pred2, y, nll = clf2.predict(x_mat, y)
    clf3 = EmbeddingUpdater(rank=1, n_ents=n_ents, n_slots=2, reg=1.0, max_epochs=500, verbose=False,
                            preprocessing=multitask_to_tuples)
    clf3.logistic2embeddings(coefs=clf1.coef_, intercept=clf1.intercept_)
    pred3, y, nll = clf3.predict(x_mat, y)

    if verbose:
        print('linear_model.LogisticRegression predictions:', prob[0:n_show])
        print('logistic regression prediction: ', 1.0 / (np.exp(-scores[0:n_show]) + 1.0))
        print('EmbeddingUpdater predictions:', 1.0 / (np.exp(-pred2[0:n_show]) + 1.0))
        print('same:', 1.0 / (np.exp(-pred3[0:n_show]) + 1.0))
    assert(np.linalg.norm(scores - pred2) < 1e-2)
    assert(np.linalg.norm(scores - pred3) < 1e-2)


if __name__ == "__main__":
    for v in [False, True]:
        test_logistic_regression_equivalence_prediction(verbose=v)
        test_logistic_regression_equivalence_learning(verbose=v)
