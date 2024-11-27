import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aug import random_aug
from dataset import load
from model import ECRGS, LogReg
from utils import sim, edge_loss, evaluate_clustering, compute_edge_embeddings, linear_transform, nes_clustering
from args import parse_args
from sklearn.model_selection import StratifiedKFold
import warnings
from seed import set_env

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

args = parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

set_env(args.seed)

if __name__ == '__main__':

    print(args)
    graph, feat, labels, num_class = load(args.dataname)
    in_dim = feat.shape[1]
    N = graph.number_of_nodes()
    model = ECRGS(in_dim, args.hid_dim, args.out_dim, args.n_layers, N, num_proj_hidden=args.proj_dim, tau=args.tau)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    graph_cuda = graph.to(args.device)
    graph_cuda = graph_cuda.remove_self_loop().add_self_loop()
    feat_cuda = feat.to(args.device)

    for epoch in range(1, args.epoch1 + 1):
        model.train()
        optimizer.zero_grad()

        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)
        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        z1, z2, z = model(graph1, feat1, graph2, feat2, graph_cuda, feat_cuda)

        edges1 = graph1.remove_self_loop().edges()
        edges2 = graph2.remove_self_loop().edges()

        indices1 = th.stack(edges1)
        values1 = th.ones(indices1.size(1), dtype=th.int).to(args.device)
        adj1 = th.sparse_coo_tensor(indices=indices1, values=values1, size=(N, N)).to(args.device)

        indices2 = th.stack(edges2)
        values2 = th.ones(indices2.size(1), dtype=th.int).to(args.device)
        adj2 = th.sparse_coo_tensor(indices=indices2, values=values2, size=(N, N)).to(args.device)

        ew_1 = model.edgeattention(z1, adj1)
        ew_2 = model.edgeattention(z2, adj2)

        edges1 = graph1.edges()
        edge_embeddings1 = compute_edge_embeddings(edges1, z1, ew_1)

        edges2 = graph2.edges()
        edge_embeddings2 = compute_edge_embeddings(edges2, z2, ew_2)

        probs_z1, probs_z2, probs_e1, probs_e2, cluster_adj_z1, cluster_adj_z2, sloid_cluster_labels_z1, sloid_cluster_labels_z2, sloid_cluster_labels_e1, sloid_cluster_labels_e2 = nes_clustering(
            z, z1, z2, graph1.edges(),
            graph2.edges(), args.nclusters,
            args.niter, args.sigma,
            edge_embeddings1, edge_embeddings2)

        ew_1 = ew_1 + args.zen
        ew_2 = ew_2 + args.zen

        con_uu = (ew_1 + ew_1.t()) * (sim(probs_z1, probs_z1) + sim(z1, z1))
        con_vv = (ew_2 + ew_2.t()) * (sim(probs_z2, probs_z2) + sim(z2, z2))

        ew_3 = (ew_1 / ew_2) * args.t1
        con_uv = (ew_3 + ew_3.t()) * (sim(probs_z1, probs_z2) + sim(z1, z2))

        dis_uu = args.t2 - con_uu
        dis_vv = args.t2 - con_vv
        dis_uv = args.t2 - con_uv

        target_min, target_max = args.tmin, args.tmax
        min_dis_uu, max_dis_uu = dis_uu.min(), dis_uu.max()
        min_dis_vv, max_dis_vv = dis_vv.min(), dis_vv.max()
        min_dis_uv, max_dis_uv = dis_uv.min(), dis_uv.max()

        dis_uu = linear_transform(dis_uu, min_dis_uu, max_dis_uu, target_min, target_max)
        dis_vv = linear_transform(dis_vv, min_dis_vv, max_dis_vv, target_min, target_max)
        dis_uv = linear_transform(dis_uv, min_dis_uv, max_dis_uv, target_min, target_max)

        conloss = model.con_loss(z1, adj1, z2, adj2, cluster_adj_z1, cluster_adj_z2, con_uu, con_vv, con_uv, dis_uu,
                                 dis_vv,
                                 dis_uv, args.hc)

        edgeloss = edge_loss(graph1.edges(), graph2.edges(), sloid_cluster_labels_z1, sloid_cluster_labels_z2,
                             sloid_cluster_labels_e1, sloid_cluster_labels_e2, probs_z1, probs_z2, probs_e1, probs_e2,
                             con_uu, con_vv)

        loss = conloss + args.alpha * edgeloss

        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Node classification evaluation ===")
    embeds = model.get_embedding(graph_cuda, feat_cuda)
    labels = th.tensor(labels)
    label = labels.to(args.device)
    skf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.splits_seed)
    test_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeds, labels)):
        train_embs = embeds[train_idx]
        test_embs = embeds[test_idx]

        train_labels = label[train_idx]
        test_labels = label[test_idx]

        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

        logreg = logreg.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
        fold_acc = 0

        for epoch in range(1, args.epoch2 + 1):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                test_logits = logreg(test_embs)
                test_preds = th.argmax(test_logits, dim=1)
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
                fold_acc = max(fold_acc, test_acc)
        print('Fold: {}, fold_acc: {:.4f}'.format(fold + 1, fold_acc))
        test_accs.append(fold_acc.item())

    avg_test_acc = sum(test_accs) / args.num_splits
    print("Average test Accuracy:", avg_test_acc)

    if args.clustering:
        print("=== Node clustering evaluation ===")
        nmi, nmi_std, ari, ari_std = evaluate_clustering(embeds, num_class, labels, args.repetition_cluster)
        print('nmi:{:.4f}, nmi std:{:.4f}, ari:{:.4f}, ari std:{:.4f}'.format(nmi, nmi_std, ari, ari_std))
