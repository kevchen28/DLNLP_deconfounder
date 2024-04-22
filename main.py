import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
from models.netdeconf import GCN_DECONF
import utils
import csv

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--extrastr', type=str, default='')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument('--nout', type=int, default=2)
parser.add_argument('--nin', type=int, default=2)
parser.add_argument('--tr', type=float, default=0.6)
parser.add_argument('--path', type=str, default='./datasets/')
parser.add_argument('--normy', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss().to(device)
alpha = torch.tensor([args.alpha], device=device, dtype=torch.float32)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def prepare(i_exp):
    # Load data and init models
    X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(i_exp), extra_str=args.extrastr)
    n = X.shape[0]
    n_train = int(n * args.tr)
    n_test = int(n * 0.2)
    # n_valid = n_test

    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    X = utils.normalize(X) #row-normalize
    X = torch.FloatTensor(X.toarray()).to(device)
    Y1 = torch.FloatTensor(np.squeeze(Y1)).to(device)
    Y0 = torch.FloatTensor(np.squeeze(Y0)).to(device)
    T = torch.LongTensor(np.squeeze(T)).to(device)

    edge_index, edge_weight = utils.convert_sparse_matrix_to_edge_list(A)
    edge_index = torch.LongTensor(edge_index).to(device)
    edge_weight = torch.FloatTensor(edge_weight).to(device)

    data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=torch.stack((Y0, Y1), dim=1), t=T)

    model = GCN_DECONF(nfeat=X.shape[1], nhid=args.hidden, dropout=args.dropout, n_out=args.nout, n_in=args.nin, cuda=args.cuda).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return data, idx_train, idx_val, idx_test, model, optimizer

def train(epoch, data, idx_train, idx_val, model, optimizer):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(data)
    yf_pred, rep, p1 = output

    print("Shape of rep:", rep.shape)  # Debugging size of representations
    print("Treatment tensor (T):", data.t[idx_train])  # Inspect treatment tensor


    rep_t1, rep_t0 = rep[idx_train][(data.t[idx_train] > 0).nonzero(as_tuple=True)], rep[idx_train][(data.t[idx_train] < 1).nonzero(as_tuple=True)]
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)

    YF = torch.where(data.t > 0, data.y[:, 0], data.y[:, 1])
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]

    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    loss_train.backward()
    optimizer.step()

    if epoch % 10 == 0:
        loss_val = loss(yf_pred[idx_val], YFva) + alpha * dist

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

def eva(data, idx_train, idx_test, model, args):
    model.eval()
    # Propagate the data through the model
    output = model(data)
    yf_pred, rep, p1 = output  # p1 can be used as propensity scores
    ycf_pred, _, _ = model(data.x, data.edge_index, 1 - data.t)

    # Get the actual outcomes from the data object
    YF = torch.where(data.t > 0, data.y[:, 0], data.y[:, 1])
    YCF = torch.where(data.t > 0, data.y[:, 1], data.y[:, 0])

    # Normalize the outputs if required
    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        y1_pred, y0_pred = (yf_pred * ys + ym), (ycf_pred * ys + ym)
    else:
        y1_pred, y0_pred = yf_pred, ycf_pred

    # Calculate PEHE
    pehe_ts = torch.sqrt(torch.mean((y1_pred - y0_pred)[idx_test] - (YF - YCF)[idx_test])**2)
    # Calculate MAE of ATE
    mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test]) - torch.mean((YF - YCF)[idx_test]))

    print("Test set results:",
          "pehe_ts= {:.4f}".format(pehe_ts.item()),
          "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))

    # Handle output file path and write results
    of_path = './new_results/' + args.dataset + args.extrastr + '/' + str(args.tr)
    of_path += ('lr'+str(args.lr) if args.lr != 1e-2 else '') + \
               ('hid'+str(args.hidden) if args.hidden != 100 else '') + \
               ('do'+str(args.dropout) if args.dropout != 0.5 else '') + \
               ('ep'+str(args.epochs) if args.epochs != 50 else '') + \
               ('lbd'+str(args.weight_decay) if args.weight_decay != 1e-5 else '') + \
               ('nout'+str(args.nout) if args.nout != 1 else '') + \
               ('alp'+str(args.alpha) if args.alpha != 1e-5 else '') + \
               ('normy' if args.normy == 1 else '') + '.csv'

    with open(of_path, 'a') as of:
        wrt = csv.writer(of)
        wrt.writerow([pehe_ts.item(), mae_ate_ts.item()])


if __name__ == '__main__':
    for i_exp in range(0,10):
        data, idx_train, idx_val, idx_test, model, optimizer = prepare(i_exp)
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch, data, idx_train, idx_val, model, optimizer)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        eva(data, idx_train, idx_test, model, args)
