import contextlib
import glob
import math
import os

import numpy as np
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import models
import utils

logger = utils.get_logger()

def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim

class Trainer(object):
    """A class to wrap training code."""
    def __init__(self,
                 dataset,
                 n_tranformers,
                 n_scalers,
                 n_constructers,
                 n_selecters,
                 n_models,
                 lstm_size,
                 temperature,
                 tanh_constant,
                 save_dir,
                 func_names,
                 controller_max_step=100,
                 controller_grad_clip=0,
                 optimizer='sgd',
                 controller_lr=0.001,
                 entropy_weight=0.001,
                 ema_baseline_decay=0.95,
                 use_tensorboard=True,
                 model_dir=None,
                 log_step=10):

        self.dataset =  dataset
        self.controller_max_step = controller_max_step
        self.controller_grad_clip = controller_grad_clip
        self.n_tranformers = n_tranformers
        self.n_scalers = n_scalers
        self.n_constructers = n_constructers
        self.n_selecters = n_selecters
        self.n_models = n_models
        self.lstm_size = lstm_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.controller_lr = controller_lr
        self.entropy_weight = entropy_weight
        self.ema_baseline_decay = ema_baseline_decay
        self.func_names = func_names
        self.use_tensorboard = use_tensorboard
        self.log_step = log_step
        self.model_dir = model_dir

        if self.use_tensorboard:
            self.tb = TensorBoard(self.model_dir)
        else:
            self.tb = None

        self.controller_step = 0

    def get_reward(self, actions):
        reward = models.fit(actions, self.dataset)
        return reward

    def random_actions(self):
        num_tokens = [self.n_tranformers, self.n_scalers,self.n_constructers, self.n_selecters, self.n_models]
        skip_index = [np.random.randint(i, size=1) for i in range(1,5)]
        func_index = [np.random.randint(i, size=1) for i in num_tokens]
        actions = []
        for x in range(4):
            actions.append(skip_index[x][0])
            actions.append(func_index[x][0])
        actions.append(func_index[-1][0])
        return actions

    def train_controller(self):

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        controller = models.Controller(self.n_tranformers,
                     				   self.n_scalers,
                                       self.n_constructers,
    	             				   self.n_selecters,
    					               self.n_models,
                                       self.func_names,
    					               self.lstm_size,
    					               self.temperature,
    					               self.tanh_constant,
    					               self.save_dir)

        controller_optimizer = _get_optimizer(self.optimizer)
        controller_optim = controller_optimizer(
            controller.parameters(),
            lr=self.controller_lr)
    	
        controller.train()
        total_loss = 0

        results_dag = []
        results_acc = []
        random_history = []
        acc_history = []

        for step in range(self.controller_max_step):
            # sample models
            dags, actions, sample_entropy, sample_log_probs = controller()
            sample_entropy =  torch.sum(sample_entropy)
            sample_log_probs = torch.sum(sample_log_probs)
            # print(sample_log_probs)
            print(actions)

            random_actions = self.random_actions() 
            with torch.no_grad():
                acc = self.get_reward(actions)
                random_acc = self.get_reward(torch.LongTensor(random_actions))

            random_history.append(random_acc)
            results_acc.append(acc)
            results_dag.append(dags)
            acc_history.append(acc)

            rewards =  torch.tensor(acc)

            if self.entropy_weight is not None:
                rewards += self.entropy_weight * sample_entropy

            reward_history.append(rewards)
            entropy_history.append(sample_entropy)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.append(adv)

             # policy loss
            loss = sample_log_probs*adv


            # update
            controller_optim.zero_grad()
            loss.backward()

            if self.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(controller.parameters(),
                                              self.controller_grad_clip)
            controller_optim.step()

            total_loss += loss.item()

            if ((step % self.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                 adv_history,
                                                 entropy_history,
                                                 reward_history,
                                                 acc_history,
                                                 random_history,
                                                 avg_reward_base,
                                                 dags)

                reward_history, adv_history, entropy_history,acc_history,random_history = [], [], [],[],[]
                total_loss = 0
            self.controller_step += 1

        max_acc = np.max(results_acc)
        max_dag = results_dag[np.argmax(results_acc)]
        path = os.path.join(self.model_dir, 'networks', 'best.png')
        utils.draw_network(max_dag[0], path)
        # np.sort(results_acc)[-10:]
        return  np.sort(list(set(results_acc)))[-10:]

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    acc_history,
                                    random_history,
                                    avg_reward_base,
                                    dags):
            """Logs the controller's progress for this training epoch."""
            cur_loss = total_loss / self.log_step

            avg_adv = np.mean(adv_history)
            avg_entropy = np.mean(entropy_history)
            avg_reward = np.mean(reward_history)
            avg_acc = np.mean(acc_history)
            avg_random = np.mean(random_history)

            if avg_reward_base is None:
                avg_reward_base = avg_reward

            logger.info(
                f'| lr {self.controller_lr:.5f} '
                f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
                f'| loss {cur_loss:.5f}')

            # Tensorboard
            if self.tb is not None:
                self.tb.scalar_summary('controller/loss',
                                       cur_loss,
                                       self.controller_step)
                self.tb.scalar_summary('controller/reward',
                                       avg_reward,
                                       self.controller_step)
                self.tb.scalar_summary('controller/reward-B_per_epoch',
                                       avg_reward - avg_reward_base,
                                       self.controller_step)
                self.tb.scalar_summary('controller/entropy',
                                       avg_entropy,
                                       self.controller_step)
                self.tb.scalar_summary('controller/adv',
                                       avg_adv,
                                       self.controller_step)
                self.tb.scalar_summary('controller/acc',
                                       avg_acc,
                                       self.controller_step)
                self.tb.scalar_summary('controller/random',
                                       avg_random,
                                       self.controller_step)

                paths = []
                # for dag in dags:
                #     fname = (f'{self.controller_step:06d}-'
                #              f'{avg_reward:6.4f}.png')
                #     path = os.path.join(self.model_dir, 'networks', fname)
                #     utils.draw_network(dag, path)
                #     paths.append(path)

                # self.tb.image_summary('controller/sample',
                #                       paths,
                #                       self.controller_step)