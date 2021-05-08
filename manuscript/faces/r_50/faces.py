import argparse
import os
import sys
import warnings

parser = argparse.ArgumentParser(description = "")
parser.add_argument("--srcpath", metavar = "srcpath", type = str) 
parser.add_argument("--split", metavar = "split", type = int, default = 5) # training split 
parser.add_argument("--srand", metavar = "srand", type = int, default = 0) # random seed 
parser.add_argument("--r", metavar = "r", type = int, default = 10) # rank 
parser.add_argument("--init", metavar = "init", type = str, default = "svd") # init 
parser.add_argument("--lamda", metavar = "lamda", type = float, default = 10)
parser.add_argument("--rho0", metavar = "rho0", type = float, default = 0.005) 
parser.add_argument("--eps", metavar = "eps", type = float, default = 1e-3)
parser.add_argument("--lr", metavar = "lr", type = float, default = 1)
parser.add_argument("--tol", metavar = "tol", type = float, default = 1e-3)
parser.add_argument("--n_iter", metavar = "n_iter", type = int, default = 25)
parser.add_argument("--outfile", metavar = "outfile", type = str, default = "output")

args = parser.parse_args()

import numpy as np
import copy
import tensorly as tl
from tensorly import tenalg, decomposition, cp_tensor
from tensorly.contrib.sparse import tensor as sptensor
import ot
import torch
import sklearn 
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat

tl.set_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl_dtype = tl.float64

sys.path.insert(0, args.srcpath)
import wtf

# load Olivetti faces dataset
# simple image scaling to (nR x nC) size
def scale(im, nR, nC):
    nR0 = len(im)     # source number of rows 
    nC0 = len(im[0])  # source number of columns 
    return np.array([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
             for c in range(nC)] for r in range(nR)])

data = sklearn.datasets.fetch_olivetti_faces()
sizex, sizey = (32, 32)
X = tl.tensor(np.array([wtf.normalise(scale(i, sizex, sizey)) for i in data.images]), dtype = tl_dtype)
target = data.target

# set random seed 
np.random.seed(args.srand)
# code to subsample randomly
train_mask = np.array([np.isin(np.arange(10), np.random.choice(10, args.split, replace = False)) for _ in range(40)])
train_idx = (np.arange(len(target)).reshape(40, 10)[train_mask]).reshape(-1)
test_idx = (np.arange(len(target)).reshape(40, 10)[~train_mask]).reshape(-1)
X_train = X[train_idx, :, :]
X_test = X[test_idx, :, :]

# setup cost matrix
xx, yy = np.meshgrid(range(sizex), range(sizey))
coords = np.vstack((xx.reshape(1, sizex*sizey), yy.reshape(1, sizex*sizey))).T
C_full = ot.utils.euclidean_distances(coords, coords, squared=True)
C_full = torch.Tensor(C_full/C_full.mean()).to(device)

# setup WTF problem 
d = 3
r = [args.r, ]*3
S = tl.zeros(r).to(device)
for i in range(r[0]):
    S[i, i, i] = 1

# initialise using SVD components as done by non_negative_parafac, hence n_iter_max = 0
factor_cp = tl.decomposition.non_negative_parafac(X_train, rank = r[0], n_iter_max = 0, init = args.init)
A = copy.deepcopy(factor_cp.factors)
A = [a.to(device) for a in A]
X0_train = X_train.to(device)

lr = np.array([np.ones(3), ]*args.n_iter)*args.lr
lamda = np.array([np.ones(3), ]*args.n_iter)*args.lamda
rho = np.array([np.ones(3)*args.rho0, ]*args.n_iter)/r[0]
eps = np.ones((args.n_iter, 3))*args.eps
# initial factor matrices need to be normalised 
A[0] = (A[0].T/A[0].sum(1)).T
A[1] = A[1]/A[1].sum(0)
A[2] = A[2]/A[2].sum(0)

dual_objs = [[], [], [], ] 

max_iter, print_inter, check_iter, unbal = (250, 10, 10, True) 
mode = "lbfgs"
for i in range(args.n_iter):
    print("Block iteration ", i)
    print("Mode 0")
    m0 = wtf.FactorsModel(X0_train, 0, [C_full, ], S, A, rho[i, :], eps[i, :], lamda[i, :], ot_mode = "slice", U_init = None, device = device, unbal = unbal, norm = "row")
    dual_objs[0] += [wtf.solve(m0, lr = lr[i, 0], mode = mode, max_iter = max_iter, print_inter = print_inter, check_iter = check_iter, tol = args.tol), ]
    A[0] = m0.compute_primal_variable().detach()
    print("Mode 1") 
    m1 = wtf.FactorsModel(X0_train, 1, [C_full, ], S, A, rho[i, :], eps[i, :], lamda[i, :], ot_mode = "slice", U_init = None, device = device, unbal = unbal, norm = "col")
    dual_objs[1] += [wtf.solve(m1, lr = lr[i, 1], mode = mode, max_iter = max_iter, print_inter = print_inter, check_iter = check_iter, tol = args.tol), ]
    A[1] = m1.compute_primal_variable().detach()
    print("Mode 2") 
    m2 = wtf.FactorsModel(X0_train, 2, [C_full, ], S, A, rho[i, :], eps[i, :], lamda[i, :], ot_mode = "slice", U_init = None, device = device, unbal = unbal, norm = "col")
    dual_objs[2] += [wtf.solve(m2, lr = lr[i, 2], mode = mode, max_iter = max_iter, print_inter = print_inter, check_iter = check_iter, tol = args.tol), ]
    A[2] = m2.compute_primal_variable().detach()

X_hat = tl.tenalg.multi_mode_dot(S, A).cpu()
factor_cp = tl.decomposition.non_negative_parafac(X_train, rank = r[0], init = "svd", n_iter_max = 500)
X_cp = tl.cp_tensor.cp_to_tensor(factor_cp)

from sklearn import cluster
kmeans = sklearn.cluster.KMeans(n_clusters = 40, n_init = 100)
clust_ot = kmeans.fit_predict(A[0].cpu())
clust_cp = kmeans.fit_predict(factor_cp.factors[0])
nmi_ot = sklearn.metrics.normalized_mutual_info_score(target[train_idx], clust_ot)
nmi_cp = sklearn.metrics.normalized_mutual_info_score(target[train_idx], clust_cp)

corr = lambda x, y: 1 - np.dot(x/np.linalg.norm(x), y/np.linalg.norm(y))
classif = {}

# classification using WTF
from sklearn import svm
from sklearn import neighbors
# clf = svm.SVC()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, metric = corr)
clf.fit(A[0].cpu(), target[train_idx])
# fit new coefficients using learned basis
A_test = [tl.ones((X_test.shape[0], r[0]), dtype = tl_dtype), tl.copy(A[1]), tl.copy(A[2])]
m0 = wtf.FactorsModel(tl.tensor(X_test, dtype = tl_dtype).cuda(), 0, [C_full, ], S, A_test, rho[-1, :], eps[-1, :], lamda[-1, :], 
                                 ot_mode = "slice", U_init = None, device = device, unbal = unbal, norm = "row")
wtf.solve(m0, lr = 1, mode = "lbfgs", max_iter = 100, print_inter = 10, check_iter = 10, tol = 1e-5)
A_test[0] = m0.compute_primal_variable().detach()
err_train = (clf.predict(A[0].cpu()) != target[train_idx]).mean() # train error
err_test = (clf.predict(A_test[0].cpu()) != target[test_idx]).mean() # test error
print("Classification with WTF")
print("err_train = ", err_train, " err_test = ", err_test)
classif["ot"] = {"err_train" : err_train, "err_test" : err_test}

# classification using Frobenius-CP
from sklearn import decomposition
# clf_cp = svm.SVC()
clf_cp = neighbors.KNeighborsClassifier(n_neighbors = 1, metric = corr)
clf_cp.fit(factor_cp.factors[0], target[train_idx])
# keep basis fixed, learn coefficients w.r.t. squared Frobenius norm.
# problem is convex (in a single factor), use sklearn.decomposition.NMF to solve for coefficients.
cp_fitter = sklearn.decomposition.NMF(max_iter = 1000) 
cp_fitter.n_components_ = r[0] 
cp_fitter.components_ = np.array(tl.tenalg.khatri_rao(factor_cp.factors[1:])).astype(np.float64).T
coeffs = cp_fitter.transform(tl.unfold(X_test, 0))
err_train = (clf_cp.predict(factor_cp.factors[0]) != target[train_idx]).mean() # train error
err_test = (clf_cp.predict(coeffs) != target[test_idx]).mean() # test error
print("Classification with CP")
print("err_train = ", err_train, " err_test = ", err_test)
classif["cp"] = {"err_train" : err_train, "err_test" : err_test}

# classification using PCA (breaks the non-negativity constraint)
pca = sklearn.decomposition.PCA(n_components = r[0], svd_solver = 'full')
X_train_pca = pca.fit_transform(tl.unfold(X_train, 0))
X_test_pca = pca.transform(tl.unfold(X_test, 0))
# clf_pca = svm.SVC()
clf_pca = neighbors.KNeighborsClassifier(n_neighbors = 1, metric = corr)
clf_pca.fit(X_train_pca, target[train_idx])
err_train = (clf_pca.predict(X_train_pca) != target[train_idx]).mean() # train error
err_test = (clf_pca.predict(X_test_pca) != target[test_idx]).mean() # test error
print("Classification with PCA")
print("err_train = ", err_train, " err_test = ", err_test)
classif["pca"] = {"err_train" : err_train, "err_test" : err_test}

np.savez(args.outfile,
         train_idx = train_idx, test_idx = test_idx, target = target, 
         A = [a.cpu().numpy() for a in A], A_cp = factor_cp, 
         X_train = X_train.numpy(), X_test = X_test.numpy(),
         X_hat = X_hat.numpy(), X_cp = X_cp, 
         dual_objs = dual_objs,
         nmi = {"ot" : nmi_ot, "cp": nmi_cp},
         classif = classif
         )
