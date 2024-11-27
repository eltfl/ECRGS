import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import faiss


def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())


def nes_clustering(x, z1, z2, edge_index_z1, edge_index_z2, nclusters, niter, sigma, edge_embeddings1,
                   edge_embeddings2):
    kmeans = faiss.Kmeans(x.shape[1], nclusters, niter=niter)
    kmeans.train(x.cpu().detach().numpy())
    centroids = th.FloatTensor(kmeans.centroids).to(x.device)

    logits_z1 = []
    logits_z2 = []
    logits_e1 = []
    logits_e2 = []
    for c in centroids:
        logits_z1.append((-th.square(z1 - c).sum(1) / sigma).view(-1, 1))
        logits_z2.append((-th.square(z2 - c).sum(1) / sigma).view(-1, 1))
        logits_e1.append((-th.square(edge_embeddings1 - c).sum(1) / sigma).view(-1, 1))
        logits_e2.append((-th.square(edge_embeddings2 - c).sum(1) / sigma).view(-1, 1))
    logits_z1 = th.cat(logits_z1, axis=1)
    logits_z2 = th.cat(logits_z2, axis=1)
    logits_e1 = th.cat(logits_e1, axis=1)
    logits_e2 = th.cat(logits_e2, axis=1)

    probs_z1 = F.softmax(logits_z1, dim=1)
    probs_z2 = F.softmax(logits_z2, dim=1)
    probs_e1 = F.softmax(logits_e1, dim=1)
    probs_e2 = F.softmax(logits_e2, dim=1)

    sloid_cluster_labels_z1 = th.argmax(logits_z1, dim=1)
    cluster_adj_z1 = sloid_cluster_labels_z1.unsqueeze(0) == sloid_cluster_labels_z1.unsqueeze(1)
    cluster_adj_z1 = cluster_adj_z1.int()

    sloid_cluster_labels_z2 = th.argmax(logits_z2, dim=1)
    cluster_adj_z2 = sloid_cluster_labels_z2.unsqueeze(0) == sloid_cluster_labels_z2.unsqueeze(1)
    cluster_adj_z2 = cluster_adj_z2.int()

    sloid_cluster_labels_e1 = th.argmax(logits_e1, dim=1)
    sloid_cluster_labels_e2 = th.argmax(logits_e2, dim=1)

    return probs_z1, probs_z2, probs_e1, probs_e2, cluster_adj_z1, cluster_adj_z2, sloid_cluster_labels_z1, sloid_cluster_labels_z2, sloid_cluster_labels_e1, sloid_cluster_labels_e2


def edge_loss(edge_index_z1, edge_index_z2, sloid_cluster_labels_z1, sloid_cluster_labels_z2, sloid_cluster_labels_e1,
              sloid_cluster_labels_e2, probs_z1, probs_z2, probs_e1, probs_e2, con_uu, con_vv):

    src_z1, dst_z1 = edge_index_z1
    src_z2, dst_z2 = edge_index_z2

    positive_mask_u = (sloid_cluster_labels_z1[src_z1] == sloid_cluster_labels_z1[dst_z1]) & \
                      (sloid_cluster_labels_z1[src_z1] == sloid_cluster_labels_e1)

    one_point_1 = src_z1[positive_mask_u]
    two_point_1 = dst_z1[positive_mask_u]
    only_edge_1 = positive_mask_u.nonzero(as_tuple=True)[0]

    positive_mask_v = (sloid_cluster_labels_z2[src_z2] == sloid_cluster_labels_z2[dst_z2]) & \
                      (sloid_cluster_labels_z2[src_z2] == sloid_cluster_labels_e2)

    one_point_2 = src_z2[positive_mask_v]
    two_point_2 = dst_z2[positive_mask_v]
    only_edge_2 = positive_mask_v.nonzero(as_tuple=True)[0]

    edgeloss_u = (con_uu[one_point_1, two_point_1] * (
                F.mse_loss(probs_z1[one_point_1], probs_e1[only_edge_1], reduction='none').sum(1) + F.mse_loss(
            probs_e1[only_edge_1], probs_z1[two_point_1], reduction='none').sum(1))).mean().item()

    edgeloss_v = (con_vv[one_point_2, two_point_2] * (
                F.mse_loss(probs_z2[one_point_2], probs_e2[only_edge_2], reduction='none').sum(1) + F.mse_loss(
            probs_z2[two_point_2], probs_e2[only_edge_2], reduction='none').sum(1))).mean().item()

    edgeloss = edgeloss_u + edgeloss_v

    return edgeloss


def evaluate_clustering(emb, nb_class, true_y, repetition_cluster):
    embeddings = F.normalize(emb, dim=-1, p=2).detach().cpu().numpy()

    estimator = KMeans(n_clusters=nb_class)

    NMI_list = []
    ARI_list = []

    for _ in range(repetition_cluster):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        nmi_score = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(true_y, y_pred)
        NMI_list.append(nmi_score)
        ARI_list.append(ari_score)

    return np.mean(NMI_list), np.std(NMI_list), np.mean(ARI_list), np.std(ARI_list)


def compute_edge_embeddings(edges, z, ew):
    with th.no_grad():
        src_nodes = edges[0]
        dst_nodes = edges[1]
        src_embeddings = z[src_nodes]
        dst_embeddings = z[dst_nodes]
        edge_weights_src = ew[src_nodes, dst_nodes].view(-1, 1)
        edge_weights_dst = ew[dst_nodes, src_nodes].view(-1, 1)
        edge_embeddings = src_embeddings * edge_weights_src + dst_embeddings * edge_weights_dst
        return edge_embeddings

def linear_transform(matrix, min_val, max_val, target_min, target_max):
    with th.no_grad():
        return target_min + (matrix - min_val) / (max_val - min_val) * (target_max - target_min)
