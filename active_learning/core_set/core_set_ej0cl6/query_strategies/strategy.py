# https://github.com/ej0cl6/deep-active-learning/edit/master/query_strategies/strategy.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, loader_tr, optimizer):
        self.clf.train()    # clf: classifier   # Todo: 이거 사용하는 것 익숙하지 않음
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):    # MNIST: 156 total iter
            if batch_idx % 10 == 0:
                print("batch idx: {}".format(batch_idx))
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)   # e1: hidden layer before last layer
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def train(self):
        # init
        n_epoch = self.args['n_epoch']
        self.clf = self.net().to(self.device)
        optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])         # Todo: 왜 보통 SGD 사용할까 ... ?         #  lr: 0.01, momentum: 0.5 이면 다른건가 ... ? 왜 momentum을 0.5 로 했을까 ... ?

        # ... data
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        # 1) Todo: 왜 DataLoader 안에 handler 가 X,y 형태여도 적용이 될까 ... ?
        # 2) Todo: loader_tr 에서 loader 가 enumerate 하면 값을 뱉어낼까 ... ?
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])
        for epoch in tqdm(range(1, n_epoch + 1)):
            """
            # original: self._train(epoch, loader_tr, optimizer)
            # >> TypeError: _train() takes 3 positional arguments but 4 were given
            """
            self._train(loader_tr, optimizer)

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()

        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()

        return embedding

