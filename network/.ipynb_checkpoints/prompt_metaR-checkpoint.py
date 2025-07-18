from embedding import *
from prompt.l2p_prompt import L2P_Prompt
from prompt.coda_prompt import CODA_Prompt
from collections import OrderedDict
import torch


class RelationMetaLearner(nn.Module):
    def __init__(self, parameter, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        if parameter['l2p']:
            self.prompt = L2P_Prompt(length=parameter['length'], embed_dim=2 * embed_size,
                                 embedding_key=parameter['embedding_key'], pool_size=parameter['size'],
                                 top_k=parameter['top_k'])
        elif parameter['coda']:   
            self.prompt = CODA_Prompt(length=parameter['length'], embed_dim=2 * embed_size,
                                 embedding_key=parameter['embedding_key'], pool_size=parameter['size'],
                                 top_k=parameter['top_k'])
        
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(2 * embed_size, num_hidden1)),
            # ('bn', nn.BatchNorm1d(few + self.length * self.top_k, affine=False)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            # ('bn', nn.BatchNorm1d(few + self.length * self.top_k, affine=False)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            # ('bn', nn.BatchNorm1d(few + self.length * self.top_k, affine=False)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs, task_id, iseval=False):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x, reduce_sim = self.prompt(x, iseval, task_id)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size), reduce_sim


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class Prompt_metaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(Prompt_metaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.embedding = Embedding(dataset, parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter, parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter, parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, task_id, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        few = support.shape[1]  # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative

        rel, reduce_sim = self.relation_learner(support, task_id, iseval=iseval)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few + num_sn, -1, -1)


        # split on e1/e2 and concat on pos/neg
        sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

        p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

        y = torch.ones(p_score.shape[0], 1).to(self.device)
        self.zero_grad()
        loss = self.loss_func(p_score, n_score, y)
        loss.backward(retain_graph=True)

        grad_meta = rel.grad
        rel_q = rel - self.beta * grad_meta

        self.rel_q_sharing[curr_rel] = rel_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score, reduce_sim
