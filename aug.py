import numpy as np
import dgl

def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

def drop_feature(x, drop_prob):
    n = x.size(1)
    drop_num = int(n * drop_prob)
    idx = np.arange(n)
    np.random.shuffle(idx)
    drop_mask = idx[:drop_num]
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()
    idx = np.arange(E)
    np.random.shuffle(idx)
    mask_num = int(E * mask_prob)
    mask_idx = idx[:mask_num]
    return mask_idx
