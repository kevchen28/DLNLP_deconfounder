import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# visualize everything using tsne
from sklearn.manifold import TSNE

from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse

path = "./datasets/"
name = "BlogCatalog"
data = sio.loadmat(path + name + "/data.mat")

# setting constants
kappa1 = 10
kappa2 = 1
C = 5

if kappa2 != 0.1:
    extra_str = str(kappa2)
else:
    extra_str = ""

X = data["Attributes"]
n = X.shape[0]
X.tocoo()

A = data["Network"]
A_dense = np.array(A.todense())

LOW = 0.1

A_sparse = sparse.csr_matrix(A_dense)

# get 50 topics
lda = LatentDirichletAllocation(n_components=50)
lda.fit(X)

Z = lda.transform(X)

AZ = np.matmul(A_dense, Z)
Z = lda.transform(X)
AZ = np.matmul(A_dense, Z)

ate_list = []

# repeat the experiment 10 times
for exp_id in range(10):
    centroid_idx = random.randint(0, X.shape[0] - 1)
    Z_c1 = Z[centroid_idx, :]
    Z_c0 = np.mean(Z, axis=0)
    Z_c0 = np.mean(Z, axis=0)

    # precompute the similarity between each instance and the two centroids
    ZZ_c1 = np.matmul(Z, Z_c1)
    ZZ_c0 = np.matmul(Z, Z_c0)
    AZZ_c1 = np.matmul(AZ, Z_c1)
    AZZ_c0 = np.matmul(AZ, Z_c0)

    # compute the propensity score
    p1 = kappa1 * ZZ_c1 + kappa2 * AZZ_c1
    p0 = kappa1 * ZZ_c0 + kappa2 * AZZ_c0
    propensity = np.divide(np.exp(p1), np.exp(p1) + np.exp(p0))
    ps = pd.Series(np.squeeze(propensity))
    ps.describe()

    # visualize the propensity distribution
    ps = pd.Series(np.squeeze(propensity))
    ps.describe()

    # visualize the propensity distribution
    fig0, ax0 = plt.subplots()
    ax0.hist(propensity, bins=50)
    plt.title("propensity score distribution")
    plt.xlabel("propensity score")
    plt.ylabel("frequency")
    plt.savefig(
        "./figs/" + name + extra_str + str(exp_id) + "ps_dist.pdf", bbox_inches="tight"
    )

    # simulate treatments
    T = np.random.binomial(1, p=propensity)

    # sample noise from Gaussian
    epsilon = np.random.normal(0, 1, X.shape[0])

    # sample noise from Gaussian
    epsilon = np.random.normal(0, 1, X.shape[0])

    # simulate outcomes
    Y1 = C * (p1 + p0) + epsilon
    Y0 = C * (p0) + epsilon
    fig1, ax1 = plt.subplots()
    ax1.hist(Y1, bins=50, label="Treated")
    ax1.hist(Y0, bins=50, label="Control")
    plt.title("outcome distribution")
    plt.legend()
    plt.savefig(
        "./figs/" + name + extra_str + str(exp_id) + "outcome_dist.pdf",
        bbox_inches="tight",
    )

    # distribution of ITE
    fig2, ax2 = plt.subplots()
    ax2.hist(Y1 - Y0, bins=50, label="ITE")
    plt.title("ITE distribution")
    plt.xlabel("ITE")
    plt.ylabel("frequency")
    ax2.axvline(x=np.mean(Y1 - Y0), color="red", label="ATE")
    plt.savefig(
        "./figs/" + name + extra_str + str(exp_id) + "ite_dist.pdf", bbox_inches="tight"
    )
    ax2.legend()

    print("ATE is %.3f" % (np.mean(Y1 - Y0)))
    ate_list.append(np.mean(Y1 - Y0))

    # save the data
    # save Y1 Y0 T X
    Z_ = TSNE(n_components=2, perplexity=50, early_exaggeration=20.0).fit_transform(Z)
    labels = T  # use treatment as the binary label
    treated_idx = np.where(T == 1)[0]
    controled_idx = np.where(T == 0)[0]
    fig3, ax3 = plt.subplots()
    ax3.scatter(Z_[treated_idx, 0], Z_[treated_idx, 1], 3, marker="o", color="red")
    ax3.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3, marker="o", color="blue")

    ax3.scatter(
        np.mean(Z_[:, 0]),
        np.mean(Z_[:, 1]),
        100,
        label=r"$z_0^c$",
        marker="D",
        color="yellow",
    )
    ax3.scatter(
        Z_[centroid_idx, 0],
        Z_[centroid_idx, 1],
        100,
        label=r"$z_1^c$",
        marker="D",
        color="green",
    )
    plt.savefig("./figs/" + name + extra_str + "tsne.pdf", bbox_inches="tight")
    plt.legend(loc=2)
    plt.xlim(-100, 100)

    # get the most freq 100 words of each topic
    topics = lda.components_

    # calculate the topic 100 words in each topic
    topics_100_dims = np.argsort(topics, axis=1)[:, -100:]

    # then we get a union of all those top 100 words
    unique_100_dims = np.unique(topics_100_dims)

    # reduce the dimensions by extract the selected words
    X_100 = X[:, unique_100_dims]

    # save the data
    sio.savemat(
        "./datasets/" + name + extra_str + "/" + name + str(exp_id) + ".mat",
        {
            "X_100": X_100,
            "T": T,
            "Y1": Y1,
            "Y0": Y0,
            "Attributes": data["Attributes"],
            "Label": data["Label"],
            "Network": data["Network"],
            "Weighted_Network": A_sparse,
            "Propensity": propensity,
            "ITE": Y1 - Y0,
        },
    )

print(f"Mean ATE: {np.mean(ate_list)}")
