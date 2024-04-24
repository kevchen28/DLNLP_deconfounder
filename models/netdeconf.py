import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NetDeconf(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False):
        """
        The NetDeconf class is a PyTorch module that implements a neural network for deconfounding using graph convolutional networks.

        Args:
            nfeat (int): The number of input features.
            nhid (int): The number of hidden units.
            dropout (float): The dropout rate.
            n_in (int, optional): The number of input layers. Defaults to 1.
            n_out (int, optional): The number of output layers. Defaults to 1.
            cuda (bool, optional): Whether to use CUDA. Defaults to False.
        """
        super(NetDeconf, self).__init__()

        device = torch.device("cuda" if cuda else "cpu")

        # Create a list of GCNConv layers
        self.gc = nn.ModuleList(
            [GCNConv(nfeat, nhid)] + [GCNConv(nhid, nhid) for _ in range(n_in - 1)]
        )
        self.gc = self.gc.to(device)  # Move the layers to the device

        self.n_in = n_in  # Number of input layers
        self.n_out = n_out  # Number of output layers

        self.out_t00 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )  # Linear layer for the output of the control
        self.out_t10 = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(n_out)]).to(
            device
        )  # Linear layer for the output of the treatment
        self.out_t01 = nn.Linear(nhid, 1).to(
            device
        )  # Linear layer for the output of the control
        self.out_t11 = nn.Linear(nhid, 1).to(
            device
        )  # Linear layer for the output of the treatment

        self.dropout = dropout
        self.ps = nn.Linear(nhid, 1).to(device)  # Linear layer for the propensity score
        self.ps_prob = (
            nn.Sigmoid()
        )  # Sigmoid activation function for the propensity score

    def forward(self, data, cf=False):
        """
        The forward method is used to compute the output of the neural network.

        Args:
            data (Data): The input PyTorch Geometric data.
            cf (bool, optional): The counterfactual flag. True for counterfactuals. Defaults to False.

        Returns:
            _type_: _description_
        """
        x, adj = data.x, data.edge_index  # Get the features and the adjacency matrix

        dist = F.relu(self.gc[0](x, adj))  # Apply the first GCNConv layer
        dist = F.dropout(dist, self.dropout, training=self.training)  # Apply dropout

        for layer in range(1, self.n_in):  # Apply the rest of the GCNConv layers
            dist = F.relu(
                self.gc[layer](dist, adj)
            )  # Apply the GCNConv layer at index layer
            dist = F.dropout(
                dist, self.dropout, training=self.training
            )  # Apply dropout

        for layer in range(
            self.n_out
        ):  # Apply the output layers for the control and treatment
            y00 = F.relu(
                self.out_t00[layer](dist)
            )  # Apply the output layer for the control
            y00 = F.dropout(y00, self.dropout, training=self.training)  # Apply dropout
            y10 = F.relu(
                self.out_t10[layer](dist)
            )  # Apply the output layer for the treatment
            y10 = F.dropout(y10, self.dropout, training=self.training)  # Apply dropout

        y0 = self.out_t01(y00).view(
            -1
        )  # Apply the output layer for the control and reshape
        y1 = self.out_t11(y10).view(
            -1
        )  # Apply the output layer for the treatment and reshape

        y = (
            torch.where(data.t > 0, y1, y0)
            if not cf
            else torch.where(1 - data.t > 0, y1, y0)
        )  # Get the output for the control or treatment based on the counterfactual flag

        propensity_score = self.ps_prob(self.ps(dist)).view(
            -1
        )  # Get the propensity score

        return y, dist, propensity_score
