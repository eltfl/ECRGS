import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from utils import sim


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, edge_weight=None):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x, edge_weight=edge_weight))
        x = self.convs[-1](graph, x)

        return x


class ECRGS(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_nodes, num_proj_hidden, tau: float = 0.5):
        super(ECRGS, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim, n_layers)
        self.tau: float = tau
        self.fc1 = th.nn.Linear(out_dim, num_proj_hidden)
        self.fc2 = th.nn.Linear(num_proj_hidden, out_dim)
        self.num_nodes = num_nodes
        self.num_proj_hidden = num_proj_hidden

        self.neighboraggr = GraphConv(num_nodes, num_nodes, norm='both', weight=False, bias=False)

        self.wl = nn.Linear(out_dim, out_dim, bias=False)
        self.wr = nn.Linear(out_dim, out_dim, bias=False)
        self.a = nn.Parameter(th.Tensor(1, 2 * out_dim))
        nn.init.xavier_uniform_(self.wl.weight)
        nn.init.xavier_uniform_(self.wr.weight)
        nn.init.xavier_uniform_(self.a)

    def edgeattention(self, h, adj1):
        N = h.size(0)
        whl = self.wl(h).unsqueeze(1)
        whr = self.wr(h).unsqueeze(1)

        row, col = adj1.coalesce().indices()

        whl_selected = whl[row]
        whr_selected = whr[col]

        concat_feature = th.cat([whl_selected, whr_selected], dim=-1).squeeze(1)

        e = F.leaky_relu(th.matmul(concat_feature, self.a.t()), 0.6)

        attention_sparse = th.sparse_coo_tensor(
            indices=th.stack([row, col]),
            values=e.squeeze(),
            size=(N, N)
        )

        attention = F.softmax(attention_sparse.to_dense(), dim=1)

        return attention

    def posaug(self, graph, x, edge_weight):
        return self.neighboraggr(graph, x, edge_weight=edge_weight)

    def forward(self, graph1, feat1, graph2, feat2, graph, feat):
        z1 = self.encoder(graph1, feat1)
        z2 = self.encoder(graph2, feat2)
        z = self.encoder(graph, feat)
        return z1, z2, z

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def delete_diag(self, tar_sim):
        return tar_sim.fill_diagonal_(0)

    def high_confince(self, s, fraction):
        num_positive = th.sum(s > 0, dim=1)
        result = th.zeros_like(s)
        for i, row in enumerate(s):
            positive_indices = th.nonzero(row > 0).squeeze()
            num_to_select = max(1, int(th.ceil(num_positive[i].float() * fraction)))
            if num_positive[i] <= 1:
                result[i, positive_indices] = row[positive_indices]
            else:
                sorted_indices = th.argsort(row, descending=True)
                result[i, sorted_indices[:num_to_select]] = row[sorted_indices[:num_to_select]]
        return result

    def semi_loss(self, z1, adj1, z2, adj2, cluster_adj_z1, cluster_adj_z2, con_uu, con_uv, dis_uu, dis_uv, con_vv, hc):
        f = lambda x: th.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        hsc_con_z1 = self.high_confince(con_uu * cluster_adj_z1, hc)
        hsc_con_z2 = self.high_confince(con_vv * cluster_adj_z2, hc)

        hsc_con_z1 = hsc_con_z1.to_sparse()
        hsc_con_z2 = hsc_con_z2.to_sparse()

        with th.no_grad():
            a1 = self.sparse_logical_or(adj1, hsc_con_z1).int()
            a2 = self.sparse_logical_or(adj2, hsc_con_z2).int()

        a1_dense = a1.to_dense()
        a2_dense = a2.to_dense()

        pos = (between_sim.diag() * con_uv) + (refl_sim * a1_dense * con_uu).sum(1) + (
                    between_sim * a2_dense * con_uv).sum(1)

        dis_uu = dis_uu - dis_uu * a1_dense
        dis_uv = dis_uv - dis_uv * a2_dense

        neg_z1 = (refl_sim * dis_uu).sum(1)
        neg_z2 = (between_sim * dis_uv).sum(1)
        neg = neg_z1 + neg_z2

        loss = -th.log(pos / (pos + neg))

        return loss

    def con_loss(self, z1, graph1, z2, graph2, cluster_adj_z1, cluster_adj_z2, con_uu, con_vv, con_uv, dis_uu, dis_vv,
                 dis_uv, hc):
        if self.num_proj_hidden > 0:
            h1 = self.projection(z1)
            h2 = self.projection(z2)
        else:
            h1 = z1
            h2 = z2

        l1 = self.semi_loss(h1, graph1, h2, graph2, cluster_adj_z1, cluster_adj_z2, con_uu, con_uv, dis_uu, dis_uv,
                            con_vv, hc)
        l2 = self.semi_loss(h2, graph2, h1, graph1, cluster_adj_z2, cluster_adj_z1, con_vv, con_uv, dis_vv, dis_uv,
                            con_uu, hc)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def get_embedding(self, graph, feat):
        with th.no_grad():
            out = self.encoder(graph, feat)
            return out.detach()

    def sparse_logical_or(self, sparse_tensor1, sparse_tensor2):
        indices1, values1 = sparse_tensor1.coalesce().indices(), sparse_tensor1.coalesce().values()
        indices2, values2 = sparse_tensor2.coalesce().indices(), sparse_tensor2.coalesce().values()

        all_indices = th.cat([indices1, indices2], dim=1)
        unique_indices, _ = th.unique(all_indices, dim=1, return_inverse=True)

        new_values = th.ones(unique_indices.size(1), dtype=th.int).to(sparse_tensor1.device)
        result = th.sparse_coo_tensor(indices=unique_indices, values=new_values, size=sparse_tensor1.size())

        return result


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret
