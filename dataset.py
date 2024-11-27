import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset

import scipy.io as sio
import dgl
from dgl import DGLGraph

from torch_geometric.datasets import WikiCS


def load(name):
    if name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        a = 1
    elif name == 'wikics':
        a = 1
    else:
        raise NotImplementedError

    if name != 'cs' and name != 'wikics':
        graph = dataset[0]
        citegraph = ['citeseer', 'pubmed']
        cograph = ['photo', 'comp']
        if name in citegraph:
            a = 1
        elif name in cograph:
            N = graph.number_of_nodes()
            idx = np.arange(N)
            np.random.shuffle(idx)
    elif name == 'cs':
        net = sio.loadmat('./' + name + '.mat')
        feat, adj, labels = net['attrb'], net['network'], net['group']
        labels = labels.flatten()
        labels = one_hot_encode(labels, labels.max() + 1).astype(int)
        num_class = labels.shape[1]
    elif name == 'wikics':
        # 加载WikiCS数据集
        dataset = WikiCS(root='./WikiCS')
        data = dataset[0]
        graph = DGLGraph((data.edge_index[0], data.edge_index[1]))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        feat = graph.ndata['feat']
        labels = graph.ndata['label']
        num_class = data.y.max().item() + 1

    if name != 'cs' and name != 'wikics':
        num_class = dataset.num_classes
        feat = graph.ndata.pop('feat')
        labels = graph.ndata.pop('label')
    elif name == 'cs':
        graph = dgl.from_scipy(adj)
        feat[feat > 0] = 1
        labels = np.argmax(labels, 1)
        feat = th.FloatTensor(feat.todense())
        adj = th.tensor(adj.todense())

    return graph, feat, labels, num_class


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]
