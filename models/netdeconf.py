import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False):
        super(GCN_DECONF, self).__init__()

        device = torch.device("cuda" if cuda else "cpu")

        # Create a list of GCNConv layers
        self.gc = nn.ModuleList(
            [GCNConv(nfeat, nhid)] + [GCNConv(nhid, nhid) for _ in range(n_in - 1)]
        )
        self.gc = self.gc.to(device)

        self.n_in = n_in
        self.n_out = n_out

        self.out_t00 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )
        self.out_t10 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )
        self.out_t01 = nn.Linear(nhid, 1).to(device)
        self.out_t11 = nn.Linear(nhid, 1).to(device)

        self.dropout = dropout
        self.pp = nn.Linear(nhid, 1).to(device)
        self.pp_act = nn.Sigmoid()

    def forward(self, data, cf=False):
        x, adj = data.x, data.edge_index

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        # for layer in self.gc[1:]:
        #     rep = F.relu(layer(rep, adj))
        #     rep = F.dropout(rep, self.dropout, training=self.training)

        for layer in range(1, self.n_in):
            rep = F.relu(self.gc[layer](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for layer in range(self.n_out):
            y00 = F.relu(self.out_t00[layer](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[layer](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        # y00 = y10 = rep
        # for out00, out10 in zip(self.out_t00, self.out_t10):
        #     y00 = F.relu(out00(y00))
        #     y00 = F.dropout(y00, self.dropout, training=self.training)
        #     y10 = F.relu(out10(y10))
        #     y10 = F.dropout(y10, self.dropout, training=self.training)

        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = (
            torch.where(data.t > 0, y1, y0)
            if not cf
            else torch.where(1 - data.t > 0, y1, y0)
        )

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return y, rep, p1
