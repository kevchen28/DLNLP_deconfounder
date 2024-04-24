import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
from models.netdeconf import NetDeconf
import utils
import csv
import os
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--nocuda", type=int, default=0, help="Disables CUDA training.")
parser.add_argument("--dataset", type=str, default="BlogCatalog")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=100, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.1, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=1e-4, help="trade-off of representation balancing."
)
parser.add_argument("--clip", type=float, default=100.0, help="gradient clipping")
parser.add_argument("--nout", type=int, default=2, help="Number of output layers.")
parser.add_argument("--nin", type=int, default=2, help="Number of input layers.")
parser.add_argument("--tr", type=float, default=0.6, help="train ratio.")
parser.add_argument("--path", type=str, default="./datasets/")
parser.add_argument("--norm", type=bool, default=True, help="Normalize the outcomes.")

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
    """
    Prepare the data and model for training.

    Args:
        i_exp (int): The experiment number (used for reproducibility purposes).

    Returns:
        Data: The PyTorch Geometric data object containing the input data.
        torch.LongTensor: The indices of the training data.
        torch.LongTensor: The indices of the validation data.
        torch.LongTensor: The indices of the test data.
        torch.nn.Module: Network Deconfounder model.
        torch.optim.Optimizer: The optimizer used for training.
    """

    # Load data and init models
    X, A, T, Y1, Y0 = utils.load_data(
        args.path,
        name=args.dataset,
        original_X=False,
        exp_id=str(i_exp),
    )

    # Split data into train, validation, and test sets
    n = X.shape[0]
    n_train = int(n * args.tr)
    n_test = int(n * 0.2)

    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = (
        idx[:n_train],
        idx[n_train : n_train + n_test],
        idx[n_train + n_test :],
    )
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # Normalize the input features and convert them to PyTorch tensors
    X = utils.normalize(X)  # row-normalize
    X = torch.FloatTensor(X.toarray()).to(device)
    Y1 = torch.FloatTensor(np.squeeze(Y1)).to(device)
    Y0 = torch.FloatTensor(np.squeeze(Y0)).to(device)
    T = torch.LongTensor(np.squeeze(T)).to(device)

    edge_index, edge_weight = utils.convert_sparse_matrix_to_edge_list(A)
    edge_index = torch.LongTensor(edge_index).to(device)
    edge_weight = torch.FloatTensor(edge_weight).to(device)

    # Create a PyTorch Geometric data object
    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.stack((Y0, Y1), dim=1),
        t=T,
    )

    # Initialize the model and optimizer
    model = NetDeconf(
        nfeat=X.shape[1],
        nhid=args.hidden,
        dropout=args.dropout,
        n_out=args.nout,
        n_in=args.nin,
        cuda=args.cuda,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    return data, idx_train, idx_val, idx_test, model, optimizer


def train(epoch, data, idx_train, idx_val, model, optimizer):
    """
    Main training loop for the model.

    Args:
        epoch (int): The current epoch number.
        data (Data): The PyTorch Geometric data object containing the input data.
        idx_train (torch.LongTensor): The indices of the training data.
        idx_val (torch.LongTensor): The indices of the validation data.
        model (torch.nn.Module): Network Deconfounder model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # Propagate the data through the model
    output = model(data)
    yf_pred, rep, _ = output

    # Calculate the Wasserstein distance between the representations of treated and control units
    rep_t1, rep_t0 = (
        rep[idx_train][(data.t[idx_train] > 0).nonzero(as_tuple=True)],  # treated units
        rep[idx_train][(data.t[idx_train] < 1).nonzero(as_tuple=True)],  # control units
    )
    dist = utils.wasserstein(rep_t1, rep_t0)  # Wasserstein distance

    YF = torch.where(data.t > 0, data.y[:, 0], data.y[:, 1])  # Factual outcomes

    # Normalize the outcomes if required
    if args.norm:
        ym, ys = torch.mean(YF[idx_train]), torch.std(
            YF[idx_train]
        )  # Mean and std of YF in the training set
        YFtr, YFva = (YF[idx_train] - ym) / ys, (
            YF[idx_val] - ym
        ) / ys  # Normalized outcomes
    else:
        YFtr = YF[idx_train]  # Unnormalized outcomes
        YFva = YF[idx_val]  # Unnormalized outcomes

    # Calculate the loss
    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist  # Loss function
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # Clip the gradients
    loss_train.backward()
    optimizer.step()

    # Print the training loss every few epochs
    print_every = (
        50 if args.epochs in [200, 300, 400] else 20 if args.epochs == 100 else 100
    )
    if epoch % print_every == 0:
        loss_val = loss(yf_pred[idx_val], YFva) + alpha * dist

        print(
            "Epoch: {:4d}".format(epoch + 1),
            "Train loss: {:.4f}".format(loss_train.item()),
            "Val loss: {:.4f}".format(loss_val.item()),
        )
        
    # Save the model at the end of training
    if epoch == args.epochs - 1:
        # Save the model
        if not os.path.exists("./new_results/" + args.dataset + "/"):
            os.makedirs("./new_results/" + args.dataset + "/")
        torch.save(model.state_dict(), "./new_results/" + args.dataset + "/model.pth")


def eva(data, idx_train, idx_test, model, args):
    """
    Main evaluation loop for the model.

    Args:
        data (Data): The PyTorch Geometric data object containing the input data.
        idx_train (torch.LongTensor): The indices of the training data.
        idx_test (torch.LongTensor): The indices of the test data.
        model (torch.nn.Module): Network Deconfounder model.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the evaluation results.
    """
    # store_results_dict = {}
    model.eval()

    # Propagate the data through the model
    yf_pred, _, _ = model(data)
    ycf_pred, _, _ = model(data, cf=True)

    # Get the actual outcomes from the data object
    YF = torch.where(data.t > 0, data.y[:, 1], data.y[:, 0])
    YCF = torch.where(data.t > 0, data.y[:, 0], data.y[:, 1])

    ym, ys = torch.mean(YF[idx_train]), torch.std(
        YF[idx_train]
    )  # Mean and std of YF in the training set

    y1_pred, y0_pred = torch.where(data.t > 0, yf_pred, ycf_pred), torch.where(
        data.t > 0, ycf_pred, yf_pred
    )  # Predicted outcomes

    # Normalize the outputs if required
    if args.norm:
        y1_pred, y0_pred = (y1_pred * ys + ym), (y0_pred * ys + ym)

    # Calculate PEHE (Precision in Estimation of Heterogeneous Effects)
    # Calculate MAE of ATE (Mean Absolute Error of Average Treatment Effect)
    predicted_effects = (y1_pred - y0_pred)[idx_test]  # Predicted treatment effects
    true_effects = (data.y[:, 1] - data.y[:, 0])[idx_test]  # True treatment effects
    temp_loss = loss(
        predicted_effects, true_effects
    )  # Loss between predicted and true treatment effects

    pehe_ts = torch.sqrt(temp_loss)  # PEHE
    mae_ate_ts = torch.abs(
        torch.mean(predicted_effects) - torch.mean(true_effects)
    )  # MAE of ATE

    print(
        "Test set results:",
        "pehe_ts= {:.4f}".format(pehe_ts.item()),
        "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()),
    )

    # Handle output file path and write results
    of_path = "./new_results/" + args.dataset + "/"
    of_path += (
        ("tr" + str(args.tr))
        + ("_lr" + str(args.lr))
        + ("_hid" + str(args.hidden))
        + ("_drop" + str(args.dropout))
        + ("_ep" + str(args.epochs))
        + ("_wd" + str(args.weight_decay))
        + ("_nin" + str(args.nin))
        + ("_nout" + str(args.nout))
        + ("_alp" + str(args.alpha))
        + ".csv"
    )

    if not os.path.exists("./new_results/" + args.dataset + "/"):
        os.makedirs("./new_results/" + args.dataset + "/")

    with open(of_path, "a") as of:
        wrt = csv.writer(of)
        wrt.writerow(
            [
                args.tr,
                args.hidden,
                args.dropout,
                args.epochs,
                args.weight_decay,
                args.nin,
                args.nout,
                args.alpha,
                pehe_ts.item(),
                mae_ate_ts.item(),
            ]
        )


if __name__ == "__main__":

    final_time = time.time()

    for i_exp in range(0, 10):
        data, idx_train, idx_val, idx_test, model, optimizer = prepare(i_exp)
        t_total = time.time()

        # Train the model
        for epoch in range(args.epochs):
            train(epoch, data, idx_train, idx_val, model, optimizer)
        print("Optimization Finished!")
        print(
            f"Time elapsed for experiment {i_exp}: {np.round(time.time() - t_total, 3)}s"
        )

        # Evaluate the model
        eva(data, idx_train, idx_test, model, args)

    print(f"Total time elapsed: {time.time() - final_time}s")

    # Run all experiments
    # bash run_for_share.sh

    # Run for a single experiment
    # python main.py --tr 0.6 --path ./datasets/ --dropout 0.1 --weight_decay 1e-5 --alpha 1e-4 --lr 1e-3 --epochs 200 --dataset BlogCatalog1 --nin 1 --nout 3 --hidden 200 --clip 100.
