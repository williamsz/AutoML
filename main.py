import models
import config
import trainer
import utils

import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import warnings

import autosklearn.classification
from tpot import TPOTClassifier
import os

func_names = [
	{0:'log1p',1:'QuantileTransformer'}, {0:'StdScaler',1:'RbtScaler',2:'MinMaxScaler'},
	{0:'PolynomialFeatures',1:'FeatureAgglomeration'},
    {0:'SelectFwe',1:'SelectPercentile',2:'RFE', 3:'SelectFromModel'},
    {0:'GaussianNB',1:'RandomForestClassifier', 2:'BernoulliNB',3:'LogisticRegression',4:'DecisionTreeClassifier',5:'ExtraTreesClassifier'} 
]

def randomforest(dataset):

	X_train, X_test, y_train, y_test = dataset
	
	
	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	score = balanced_accuracy_score(y_test, y_predicted)
	# scores.append(score)
	return score

def auto_sklearn(dataset):
	X_train, X_test, y_train, y_test = dataset
	scores = []
	time = 120
	for _ in range(10):
		automl = autosklearn.classification.AutoSklearnClassifier(
		    #ensemble_size=1,
		    time_left_for_this_task=time,
		    per_run_time_limit=time//10,
		    disable_evaluator_output=False,
		    delete_tmp_folder_after_terminate=False,
		    delete_output_folder_after_terminate=False,
			)
		automl.fit(X_train, y_train)
		y_hat = automl.predict(X_test)
		score = balanced_accuracy_score(y_test, y_hat)
		scores.append(score)
	return scores

def tpot_(dataset):
	X_train, X_test, y_train, y_test = dataset
	scores = []
	for _ in range(10):
		tpot = TPOTClassifier(verbosity=2, max_time_mins=2, population_size=20,scoring='balanced_accuracy')
		tpot.fit(X_train, y_train)
		score = tpot.score(X_test, y_test)
		scores.append(score)
	return scores

def main(args):
	scores_rl = []
	scores_randomforest = []

	for _ in range(3):
		utils.prepare_dirs(args)
		utils.save_args(args)

		# max_step = range(100,1000,100)

		dataset = get_dataset(args.dataset)

		maxscore_randomforest = randomforest(dataset)
		scores_randomforest.append(maxscore_randomforest)
		# scores_autosklearn = auto_sklearn(dataset)
		# scores_tpot = tpot_(dataset)

		

		trnr = trainer.Trainer(dataset,
								args.n_tranformers,
								args.n_scalers,
								args.n_constructers,
								args.n_selecters,
								args.n_models,
								args.lstm_size,
								args.temperature,
								args.tanh_constant,
								args.save_dir,
								func_names=func_names,
								model_dir=args.model_dir,
								log_step=args.log_step,
								controller_max_step=args.controller_max_step)
		bset = trnr.train_controller()
		scores_rl.append(bset)
		print(bset)
	print(scores_rl)

	# method_names = ['RandomForest', 'AutoSklearn', 'TPOT', 'RL']
	# result_map = {'RandomForest': scores_randomforest, 'AutoSklearn':scores_autosklearn,
	# 				'TPOT': scores_tpot, 'RL': scores_rl}
	result_map = {'RandomForest': scores_randomforest, 'RL': scores_rl}
	df = pd.DataFrame(data=result_map)
	path = os.path.join(args.model_dir, 'df.csv')
	df.to_csv(path)



def get_dataset(id, RANDOM_STATE = 42):
	
	if id == 1:
		dataset = load_breast_cancer()
		X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)
	elif id == 2:
		df = pd.read_csv('/home/zhou/Downloads/UCI/winequality/winequality-red.csv',delimiter=';')
		datanp = df.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)
	elif id == 3:
		df = pd.read_csv('/home/zhou/Downloads/UCI/winequality/winequality-white.csv',delimiter=';')
		datanp = df.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)

	elif id == 4:
		df=pd.read_csv('/home/zhou/Downloads/UCI/hill-valley/hill-valley-with-noise-training.data')
		datanp = df.values
		X_train, y_train = datanp[:,:-1], datanp[:,-1].astype(int)
		df=pd.read_csv('/home/zhou/Downloads/UCI/hill-valley/hill-valley-with-noise-testing.data')
		datanp = df.values
		X_test, y_test = datanp[:,:-1], datanp[:,-1].astype(int)

	elif id == 5:
		df=pd.read_csv('/home/zhou/Downloads/UCI/spambase/spambase.data',header=None)
		datanp = df.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)

	elif id == 6:
		df=pd.read_csv('/home/zhou/Downloads/UCI/hill-valley/ionosphere',header=None)
		df[34].loc[df[34]=='g']=0
		df[34].loc[df[34]=='b']=1
		datanp = df.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)
	elif id == 7:
		df=pd.read_csv('/home/zhou/Downloads/UCI/hill-valley/glass',header=None)
		datanp = df.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)
	elif id == 8:
		dataset = load_digits()
		X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                    random_state=RANDOM_STATE)
	elif id == 9:

		diabetes=pd.read_csv('/home/zhou/Downloads/UCI/hill-valley/diabetes.csv')
		diabetes["class"].loc[diabetes["class"]=='tested_positive']=0
		diabetes["class"].loc[diabetes["class"]=='tested_negative']=1
		datanp=diabetes.values
		X_train, X_test, y_train, y_test = train_test_split(datanp[:,:-1], datanp[:,-1].astype(int),
                                                    random_state=RANDOM_STATE)
	return (X_train, X_test, y_train, y_test)

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

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)