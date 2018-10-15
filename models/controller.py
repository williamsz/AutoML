import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

import collections
import os

Node = collections.namedtuple('Node', ['id', 'name'])


def _construct_dags(actions, func_names, num_blocks=5):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.

    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.

    Returns:
        A list of DAGs defined by the inputs.

    RNN cell DAGs are represented in the following way:

    1. Each element (node) in a DAG is a list of `Node`s.

    2. The `Node`s in the list dag[i] correspond to the subsequent nodes
       that take the output from node i as their own input.

    3. dag[-1] is the node that takes input from x^{(t)} and h^{(t - 1)}.
       dag[-1] always feeds dag[0].
       dag[-1] acts as if `w_xc`, `w_hc`, `w_xh` and `w_hh` are its
       weights.

    4. dag[N - 1] is the node that produces the hidden state passed to
       the next timestep. dag[N - 1] is also always a leaf node, and therefore
       is always averaged with the other leaf nodes and fed to the output
       decoder.
    """
    dags = []
    
    prev_nodes = actions[0::2].view(1,-1)
    activations = actions[1::2].view(1,-1)
    
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # add first node
        # dag[-1] = [Node(0, func_names[func_ids[0]])]
        # dag[-2] = [Node(0, func_names[func_ids[0]])]

        dag[-1] = [Node(0, 'Data')]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids)):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[jdx][func_id.item()]))

        leaf_nodes = set(range(num_blocks)) - dag.keys()

        # merge with avg
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, func_names[-1][utils.to_item(actions[-1])])]

        # TODO(brendan): This is actually y^{(t)}. h^{(t)} is node N - 1 in
        # the graph, where N Is the number of nodes. I.e., h^{(t)} takes
        # only one other node as its input.
        # last h[t] node
        # last_node = Node(num_blocks + 1, 'h[t]')
        # dag[num_blocks] = [last_node]
        dags.append(dag)



    return dags


class Controller(torch.nn.Module):

    def __init__(self, 
                 n_tranformers,
                 n_scalers,
                 n_constructers,
                 n_selecters,
                 n_models,
                 func_names,
                 lstm_size=100,
                 temperature=None,
                 tanh_constant=None,
                 save_dir=None):

        super(Controller, self).__init__()
        self.n_tranformers = n_tranformers
        self.n_scalers = n_scalers
        self.n_constructers = n_constructers
        self.n_selecters = n_selecters
        self.n_models = n_models
        self.lstm_size = lstm_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.save_dir =  save_dir
        self.func_names = func_names

        self.lstm = torch.nn.LSTMCell(self.lstm_size, self.lstm_size)
        self.linear_1 = torch.nn.Linear(self.lstm_size, self.lstm_size)
        self.linear_2 = torch.nn.Linear(self.lstm_size, self.lstm_size)
        self.linear_3 = torch.nn.Linear(self.lstm_size, 1)
        self.emb = torch.nn.Embedding(1, self.lstm_size)

        self.num_tokens = [self.n_tranformers, self.n_scalers,self.n_constructers, self.n_selecters, self.n_models]
        self.decoders = []
        self.embeddings = []
        for i in self.num_tokens:
            decoder =  torch.nn.Linear(self.lstm_size, i)
            embedding = torch.nn.Embedding(i, self.lstm_size)
            self.decoders.append(decoder)
            self.embeddings.append(embedding)

        for i in self.num_tokens[:-1]:
            embedding = torch.nn.Embedding(i, self.lstm_size)
            self.embeddings.append(embedding)

        self._decoders = torch.nn.ModuleList(self.decoders)
        self._embeddings = torch.nn.ModuleList(self.embeddings)


    def forward(self):

        arc_seq = []
        sample_log_probs = []
        sample_entropy = []
        all_h = []
        all_h_w = []
        #size of [1, self.lstm_size]
        inputs = self.emb(torch.LongTensor([0]))
        prev_c = torch.zeros(1, self.lstm_size)
        prev_h = torch.zeros(1, self.lstm_size)
        for layer_id in range(len(self.num_tokens)-1):
            next_c, next_h = self.lstm(inputs, (prev_c, prev_h))
            prev_c, prev_h = next_c, next_h
            
            all_h.append(next_h)
            all_h_w.append(self.linear_1(next_h))

            query = self.linear_2(next_h)
            query = query + torch.cat(all_h_w, 0)
            query = F.tanh(query)
            logits = self.linear_3(query)
            logits = logits.view(1, layer_id+1)

            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * F.tanh(logits)

            # diff = (layer_id+1 - torch.range(0, layer_id)) ** 2
            # logits -= torch.reshape(diff, (1, layer_id+1)) / 6.0

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            skip_index = torch.multinomial(probs, 1)

            selected_log_prob = log_prob.gather(
                1, utils.get_variable(skip_index, requires_grad=False))
            arc_seq.append(skip_index.squeeze(1))
            sample_log_probs.append(selected_log_prob.squeeze(1))
            sample_entropy.append(entropy.detach())

            inputs = torch.index_select(
              torch.cat(all_h, 0), 0, skip_index.squeeze(1))
            # inputs /= (0.1 + (layer_id - skip_index).type(torch.FloatTensor))

            next_c, next_h = self.lstm(inputs, (prev_c, prev_h))
            prev_c, prev_h = next_c, next_h
            logits = self._decoders[layer_id](next_h)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * F.tanh(logits)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            func = torch.multinomial(probs, 1)

            selected_log_prob = log_prob.gather(
                1, utils.get_variable(func, requires_grad=False))
            arc_seq.append(func.squeeze(1))
            sample_log_probs.append(selected_log_prob.squeeze(1))
            sample_entropy.append(entropy.detach())

            inputs = self._embeddings[layer_id](func.squeeze(1))

        #for models
        next_c, next_h = self.lstm(inputs, (prev_c, prev_h))
        prev_c, prev_h = next_c, next_h
        logits = self._decoders[-1](next_h)
        if self.temperature is not None:
            logits /= self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * F.tanh(logits)
        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * probs).sum(1, keepdim=False)
        func = torch.multinomial(probs, 1)

        selected_log_prob = log_prob.gather(
            1, utils.get_variable(func, requires_grad=False))
        arc_seq.append(func.squeeze(1))
        sample_log_probs.append(selected_log_prob.squeeze(1))
        sample_entropy.append(entropy.detach())

        arc_seq = torch.cat(arc_seq, 0)
        sample_entropy = torch.cat(sample_entropy)
        sample_log_probs = torch.cat(sample_log_probs)

        dags = _construct_dags(arc_seq, self.func_names)

        # print(arc_seq)

        # if self.save_dir is not None:
        #     for idx, dag in enumerate(dags):
        #         utils.draw_network(dag,
        #                            os.path.join(self.save_dir, f'graph{idx}.png'))

        return dags, arc_seq, sample_entropy, sample_log_probs
        