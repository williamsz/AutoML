from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np

import sklearn
from sklearn import feature_selection
from sklearn import linear_model
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import load_digits
import warnings

# from sklearn.preprocessing import FunctionTransformer

func_names = [
	{0:'log1p',1:'QuantileTransformer'}, {0:'StdScaler',1:'RbtScaler',2:'MinMaxScaler'},
	{0:'PolynomialFeatures',1:'FeatureAgglomeration'},
    {0:'SelectFwe',1:'SelectPercentile',2:'RFE', 3:'SelectFromModel'},
    {0:'GaussianNB',1:'RandomForestClassifier', 2:'BernoulliNB',3:'LogisticRegression',4:'DecisionTreeClassifier',5:'ExtraTreesClassifier'} ]


def fit(actions, dataset):
	
	X_train, X_test, y_train, y_test = dataset
	seq = {}
	#fit_transformer
	if actions[1].item() == 0:
		log1p_fit_transformer = preprocessing.FunctionTransformer()
		seq[1] = log1p_fit_transformer
	else:
		quantile_fit_transformer = preprocessing.QuantileTransformer(random_state=0)
		seq[1] = quantile_fit_transformer
	#scaler
	if actions[3].item() == 0:
		standard_scaler = preprocessing.StandardScaler()
		seq[2] = standard_scaler
	elif actions[3].item() == 1:
		robust_scaler = preprocessing.RobustScaler()
		seq[2] = robust_scaler
	else:
		min_max_scaler = preprocessing.MinMaxScaler()
		seq[2] = min_max_scaler
	#constructers
	if actions[5].item() == 0:
		seq[3] = preprocessing.PolynomialFeatures(interaction_only=True)
	else:
		seq[3] = FeatureAgglomeration(5)
	#selecter
	if actions[7].item() == 0:
		selecter = feature_selection.SelectFwe()
		seq[4] = selecter
	elif actions[7].item() == 1:
		selecter = feature_selection.SelectPercentile()
		seq[4] = selecter
	elif actions[7].item() == 2:
		selecter = feature_selection.RFE(sklearn.ensemble.ExtraTreesClassifier())
		seq[4] = selecter
	else:
		selecter = feature_selection.SelectFromModel(sklearn.ensemble.ExtraTreesClassifier(),"median")
		seq[4] = selecter
	#models
	if actions[-1].item() == 0:
		model = sklearn.naive_bayes.GaussianNB()
		seq[5] = model
	elif actions[-1].item() == 1:
		model = sklearn.ensemble.RandomForestClassifier()
		seq[5] = model
	elif actions[-1].item() == 2:
		model = sklearn.naive_bayes.BernoulliNB()
		seq[5] = model
	elif actions[-1].item() == 3:
		model = linear_model.LogisticRegression()
		seq[5] = model
	elif actions[-1].item() == 4:
		model = sklearn.tree.DecisionTreeClassifier()
		seq[5] = model
	else:
		model = sklearn.ensemble.ExtraTreesClassifier()
		seq[5] = model

	#connectivity
	transformed = {}
	#For Node 1
	transformed[1] = seq[1].fit_transform(X_train)
	#For Node 2
	if actions[2].item() == 0:
		transformed[2] = seq[2].fit_transform(X_train)
	elif actions[2].item() == 1:
		transformed[2] = seq[2].fit_transform(transformed[1])
	#For Node 3
	if actions[4].item() == 0:
		transformed[3] = seq[3].fit_transform(X_train)
	elif actions[4].item() == 1:
		transformed[3] = seq[3].fit_transform(transformed[1])
	elif actions[4].item() == 2:
		transformed[3] = seq[3].fit_transform(transformed[2])
	#For Node 4
	if actions[6].item() == 0:
		transformed[4] = seq[4].fit_transform(X_train, y_train)
	elif actions[6].item() == 1:
		transformed[4] = seq[4].fit_transform(transformed[1], y_train)
	elif actions[6].item() == 2:
		transformed[4] = seq[4].fit_transform(transformed[2], y_train)
	elif actions[6].item() == 3:
		transformed[4] = seq[4].fit_transform(transformed[3], y_train)

	#leaf nodes
	leaf_nodes = set(range(5)) - {i.item() for i in actions[0:-1:2]}
	# print(leaf_nodes)
	merge_data = np.concatenate([transformed[i] for i in leaf_nodes], axis=1)
	last_selecter = feature_selection.SelectFromModel(sklearn.ensemble.ExtraTreesClassifier(),"median")
	merge_data = last_selecter.fit_transform(merge_data, y_train)
	clf = seq[5].fit(merge_data, y_train)

	#test data
	test_transformed = {}
	#For Node 1
	test_transformed[1] = seq[1].transform(X_test)
	#For Node 2
	if actions[2].item() == 0:
		test_transformed[2] = seq[2].transform(X_test)
	elif actions[2].item() == 1:
		test_transformed[2] = seq[2].transform(test_transformed[1])
	#For Node 3
	if actions[4].item() == 0:
		test_transformed[3] = seq[3].transform(X_test)
	elif actions[4].item() == 1:
		test_transformed[3] = seq[3].transform(test_transformed[1])
	elif actions[4].item() == 2:
		test_transformed[3] = seq[3].transform(test_transformed[2])
	#For Node 4
	if actions[6].item() == 0:
		test_transformed[4] = seq[4].transform(X_test)
	elif actions[6].item() == 1:
		test_transformed[4] = seq[4].transform(test_transformed[1])
	elif actions[6].item() == 2:
		test_transformed[4] = seq[4].transform(test_transformed[2])
	elif actions[6].item() == 3:
		test_transformed[4] = seq[4].transform(test_transformed[3])

	# print([test_transformed[i].shape for i in leaf_nodes])

	merge_data = np.concatenate([test_transformed[i] for i in leaf_nodes], axis=1)
	merge_data = last_selecter.transform(merge_data)
	pred_test = clf.predict(merge_data)

	# reward = metrics.accuracy_score(y_test, pred_test)
	reward = balanced_accuracy_score(y_test, pred_test)
	# print(clf)
	# print(reward)
	# print('\nPrediction accuracy for the normal test dataset with log tranformer')
	# print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

	return reward


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = metrics.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score