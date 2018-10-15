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
import torch

import autosklearn.classification
from tpot import TPOTClassifier
import os


def get_reward(actions, dataset):
    reward = models.fit(actions, dataset)
    return reward

def random_actions(args):
    num_tokens = [args.n_tranformers, args.n_scalers,args.n_constructers, args.n_selecters, args.n_models]
    skip_index = [np.random.randint(i, size=1) for i in range(1,5)]
    func_index = [np.random.randint(i, size=1) for i in num_tokens]
    actions = []
    for x in range(4):
        actions.append(skip_index[x][0])
        actions.append(func_index[x][0])
    actions.append(func_index[-1][0])
    return actions

def main(args):
	utils.prepare_dirs(args)
	utils.save_args(args)

	dataset = get_dataset(args.dataset)
	random_results = []
	for step in range(args.controller_max_step):
		actions = random_actions(args)
		reward = get_reward(torch.LongTensor(actions), dataset)
		random_results.append(reward)
	top_scores = np.sort(list(set(random_results)))[-10:]

	result_map = {'RandomSearch': top_scores}
	df = pd.DataFrame(data=result_map)
	path = os.path.join(args.model_dir, 'df.csv')
	df.to_csv(path)


def get_dataset(id, RANDOM_STATE = 42):
	
	if id == 1:
		dataset = load_breast_cancer()
		X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
			random_state=RANDOM_STATE)
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

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)