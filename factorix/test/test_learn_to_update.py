
# import numpy as np
from sklearn import metrics
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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


def test_logistic_regression_equivalence(verbose=False, plot=False):

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)

    ###############################################################################
    # Demo path functions

    #
    n_ents = X.shape[1] + 3

    # cs = np.logspace(-2, 2, 20)
    c = 1.0
    clf1 = linear_model.LogisticRegression(C=c, penalty='l2', tol=1e-6)
    clf1.fit(X, y)
    prob = clf1.predict_proba(X)
    n_show = 3
    print('linear_model.LogisticRegression predictions:', prob[0:n_show])
    scores = X.dot(clf1.coef_.T) + clf1.intercept_
    print('logistic regression prediction: ', 1.0 / (np.exp(-scores[0:n_show]) + 1.0))

    clf2 = EmbeddingUpdater(rank=1, n_ents=n_ents, n_slots=2, reg=c, max_epochs=500, verbose=False,
                            preprocessing=multitask_to_tuples)
    clf2.logistic2embeddings(coefs=clf1.coef_, intercept=clf1.intercept_)
    pred, y, nll = clf2.predict(X[0:n_show,], y[0:n_show])
    print('EmbeddingUpdater predictions:', 1.0 / (np.exp(-pred[0:n_show]) + 1.0))

    clf2.logistic2embeddings(coefs=clf1.coef_, intercept=clf1.intercept_)
    pred, y, nll = clf2.predict(X[0:n_show,], y[0:n_show])
    print('same:', 1.0 / (np.exp(-pred[0:n_show]) + 1.0))


def test_logistic_regression_equivalence_old(verbose=False, plot=False):

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)

    ###############################################################################
    # Demo path functions

    #
    n_ents = X.shape[1] + 2

    # cs = np.logspace(-2, 2, 20)
    cs = [1.0]

    clfs_groups = [
        [linear_model.LogisticRegression(C=c, penalty='l2', tol=1e-6) for c in cs],
        [EmbeddingUpdater(1, n_ents, c, max_epochs=500, n_slots=2, verbose=False, preprocessing=multitask_to_tuples)
         for c in cs]
        ]

    print("Computing regularization path ...")
    start = datetime.now()
    coefs_ = []
    for i, clfs in enumerate(clfs_groups):
        tmp = []
        for clf in clfs:
            if i is 1:
                clf.logistic2embeddings(coefs_[0][0])
            clf.fit(X, y)
            print(clf.coef_)
            tmp.append(clf.coef_.ravel().copy())
        print("This took ", datetime.now() - start)
        coefs_.append(tmp)

    if plot:
        coefs_ = np.array(coefs_)
        plt.plot(np.log10(cs), coefs_)
        ymin, ymax = plt.ylim()
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        plt.title('Logistic Regression Path')
        plt.axis('tight')
        plt.show()
    else:

        for c, coef1, coef2 in zip(cs, coefs_[0], coefs_[1]):
            print('%7.2f: [%s] \t [%s]' % (c,
                                           ', '.join(['%5.2f' % x for x in coef1]),
                                           ', '.join(['%5.2f' % x for x in coef2])))

            # model = EmbeddingUpdater(rank, n_ents, reg, max_epochs, verbose=False)
            # model.fit(data_train)
            # pred, y, test_nll = model.predict(data_test)
            # test_auc = eval_auc(pred, y)
            # if verbose:
            #     print('reg: ', reg, 'niter: ', max_epochs, ', auc: ', test_auc, ', nll: ', test_nll)


if __name__ == "__main__":
    # n_data = 6
    # n_features = 5
    # mat = tf.constant(np.outer(range(n_data), np.ones(n_features)), dtype='int64')
    # print(tf_eval(mat))
    #
    test_logistic_regression_equivalence(False)
