# Wasserstein tensor factorisation
# Author: Stephen Zhang (syz@math.ubc.ca)
# Date: April 2021

import numpy as np
import copy
import tensorly as tl
from tensorly import tenalg, decomposition, cp_tensor
from tensorly.contrib.sparse import tensor as sptensor
import torch
from torch.autograd import grad, Variable
import math

# setup
tl.set_backend("pytorch")

# define some helper functions
def inner(A, B):
    # PyTorch inner prod that works for matrices
    return (A*B).sum()

def xlogy(x, y):
    out = x*torch.log(y)
    out[x == 0] = 0
    return out

def xy(x, y):
    out = x*y
    out[x == 0] = 0
    out[y == 0] = 0
    return out

normalise = lambda x: x/x.sum()

class OTModel(torch.nn.Module):
    def __init__(self, device = None):
        super(OTModel, self).__init__()
        self.device = torch.device("cpu") if device is None else device
        pass
    def E(self, p):
        return (xlogy(p, p) - p).sum()
    def E_star_bal(self, p):
        return torch.logsumexp(p, 0).sum()
    def E_star_unbal(self, p):
        return p.exp().sum()
    def KL(self, a, b):
        return (xlogy(a, a/b) - a + b).sum()
    def KL2(self, a, b):
        return (xlogy(a, a/b) - a).sum()
    def H_star_bal(self, G, P, eps, K):
        X = G/eps
        scale = X.max(0).values
        return eps*(-self.E(P) + torch.sum(P*(scale + torch.log(K @ torch.exp(X-scale)))))
    def get_ufrac(self, u, lamda, eps, log = False):
        if log:
            return (lamda/eps)*torch.log(lamda/(lamda - u))
        else:
            return ((lamda/eps)*torch.log(lamda/(lamda - u))).exp()
    def H_star_unbal(self, U, P, eps, lamda, K, alpha = None):
        F = self.get_ufrac(U, lamda, eps, log = True)
        scale = F.max(0).values
        if alpha is None:
            return -eps*xlogy(P, P).sum() + eps*xlogy(P, K @ (F - scale).exp()).sum() + eps*(P*(scale + 1)).sum()
        else:
            return -eps*xlogy(P, P).sum(0).dot(alpha) + eps*xlogy(P, K @ (F - scale).exp()).sum(0).dot(alpha) + eps*(P*(scale + 1)).sum(0).dot(alpha)
    def H_star_unbal_stab(self, U, P, eps, lamda, C):
        F = self.get_ufrac(U, lamda, eps, log = True)
        return -eps*xlogy(P, P).sum() + eps*P.sum() + \
                eps*xy(P, torch.logsumexp((-C/eps).reshape(C.shape[0], C.shape[0], 1) + F.reshape(1, F.shape[0], F.shape[1]), dim = 1)).sum()
    def H_unbal(self, U, P, eps, lamda, K):
        out = 0
        for i in range(U.shape[1]):
            gamma = self.get_gamma_unbal(U[:, i], P[:, i], eps, lamda, K)
            q = self.get_q_unbal(U[:, i], P[:, i], eps, lamda, K)
            alpha = self.get_alpha_unbal(U[:, i], P[:, i], eps, lamda, K)
            f = self.get_ufrac(U[:, i], lamda, eps, log = True)
            out += eps*((gamma @ f).sum() + (gamma.T @ (alpha/eps)).sum() - gamma.sum()) +  lamda*self.KL(gamma.sum(0), q)
        return out
    def get_alpha_unbal(self, u, p, eps, lamda, K):
        f = self.get_ufrac(u, lamda, eps, log = True)
        scale = f.mean()
        return eps*(torch.log(p) - (torch.log(K @ (f - scale).exp()) + scale))
    def get_q_unbal(self, u, p, eps, lamda, K):
        f = self.get_ufrac(u, lamda, eps)
        return (lamda/(lamda - u))**(lamda/eps + 1) * (K.T @ (p/(K @ f)))
    def get_gamma_unbal(self, u, p, eps, lamda, K):
        return torch.einsum("i,ij,j->ij", torch.exp(self.get_alpha_unbal(u, p, eps, lamda, K)/eps),
                                          K,
                                          self.get_ufrac(u, lamda, eps))

def sumstack(l):
    if len(l) > 0:
        return torch.stack(l).sum(0)
    else:
        return 0

class FactorsModel(OTModel):
    def get_SA(self, i):
        modes = np.arange(len(self.A))
        modes = modes[modes != i]
        return tl.unfold(tl.tenalg.multi_mode_dot(self.S, [self.A[j] for j in modes], modes = modes), i)
    def set_U(self, U):
        self.U = torch.nn.ParameterList([torch.nn.Parameter(Variable(U[i], requires_grad = True)) for i in range(len(self.X_orig.shape))])
    def __init__(self, X, k, C, S, A, rho, eps, lamda, optim_modes, ot_mode = "fiber", C_full = None, U_init = None, unbal = True, norm = None, **kwargs):
        super(FactorsModel, self).__init__(**kwargs)
        self.X = [tl.unfold(X, i) for i in range(len(X.shape))] # all unfoldings
        self.alpha = [(self.X[i].sum(0) > 0)*1.0 for i in range(len(X.shape))]
        self.X_orig = X # folded tensor
        self.k = k # mode
        self.C = C
        self.unbal = unbal
        self.norm = norm
        if C is not None:
            self.K = [(-C[k]/eps[k]).exp() for k in range(len(C))] # Gibbs kernels for each mode
        if C_full is not None:
            self.C_full = C_full
            self.K_full = (-C_full/eps[0]).exp()
        self.eps = eps
        self.rho = rho
        self.lamda = lamda
        self.A = A
        self.optim_modes = optim_modes
        if U_init is None:
            U = [torch.zeros(self.X[i].shape).to(self.device) for i in range(len(X.shape))]
        else:
            U = U_init
        self.S = S
        self.S_A = self.get_SA(self.k)
        self.set_U(U)
        self.ot_mode = ot_mode
    def get_xi(self, i):
        modes = [j for j in range(len(self.A)) if j not in (i, self.k)]
        modes2 = [j for j in range(len(self.A)) if j not in (i, )]
        s = tl.tenalg.multi_mode_dot(self.S, [self.A[j] for j in modes2], modes = modes2).shape
        return tl.unfold(tl.fold(self.A[i].T @ self.U[i], i, s), self.k) @ tl.unfold(tl.tenalg.multi_mode_dot(self.S, [self.A[j] for j in modes], modes), self.k).T
    def dual_obj(self):
        if self.ot_mode == "fiber":
            return torch.sum(torch.stack([self.H_star_unbal(self.U[i], self.X[i], self.eps[i], self.lamda[i], self.K[i], alpha = self.alpha[i]) for i in self.optim_modes])) + \
                  self.rho[self.k]*self.E_star_unbal((-1/self.rho[self.k]) * (self.U[self.k] @ self.S_A.T * float(self.k in self.optim_modes) + sumstack([self.get_xi(j) for j in self.optim_modes if j != self.k])))
        elif self.ot_mode == "slice":
            ot_term = self.H_star_unbal(self.U[0].T, self.X[0].T, self.eps[0], self.lamda[0], self.K_full) if self.unbal else self.H_star_bal(self.U[0].T, self.X[0].T, self.eps[0], self.K_full)
            Z = (-1/self.rho[self.k]) * (self.U[self.k] @ self.S_A.T * float(self.k in self.optim_modes) + sumstack([self.get_xi(j) for j in self.optim_modes if j != self.k]))
            if self.norm is None:
                ent_term = self.rho[self.k]*self.E_star_unbal(Z)
            elif self.norm == "col":
                ent_term = self.rho[self.k]*self.E_star_bal(Z)
            elif self.norm == "row":
                ent_term = self.rho[self.k]*self.E_star_bal(Z.T)
            return ot_term + ent_term
            # if self.unbal:
            #     return self.H_star_unbal(self.U[0].T, self.X[0].T, self.eps[0], self.lamda[0], self.K_full) +  \
            #           self.rho[self.k]*self.E_star_unbal((-1/self.rho[self.k]) * (self.U[self.k] @ self.S_A.T * float(self.k in self.optim_modes) + sumstack([self.get_xi(j) for j in self.optim_modes if j != self.k])))
            # else:
            #     return self.H_star_bal(self.U[0].T, self.X[0].T, self.eps[0], self.K_full) +  \
            #           self.rho[self.k]*self.E_star_bal((-1/self.rho[self.k]) * (self.U[self.k] @ self.S_A.T * float(self.k in self.optim_modes) + sumstack([self.get_xi(j) for j in self.optim_modes if j != self.k])))
    def primal_obj(self, terms = False):
        # not implemented 
        return 0
    def forward(self):
        return self.dual_obj()
    def compute_primal_variable(self):
        x = self.U[self.k] @ self.S_A.T * float(self.k in self.optim_modes)
        l = [self.get_xi(j) for j in self.optim_modes if j != self.k]
        if len(l) > 0:
            x += torch.stack(l).sum(0)
        Z = (-1/self.rho[self.k])*x
        if self.norm is None:
            return torch.exp((-1/self.rho[self.k])*x)
        elif self.norm == "col":
            a = torch.exp((-1/self.rho[self.k])*x)
            return a/a.sum(0).reshape(1, -1)
        elif self.norm == "row":
            a = torch.exp((-1/self.rho[self.k])*x)
            return (a.T/a.sum(1)).T
#
class CoreModel(OTModel):
    def set_U(self, U):
        self.U = torch.nn.ParameterList([torch.nn.Parameter(Variable(U[i], requires_grad = True)) for i in range(len(self.X_orig.shape))])
    def __init__(self, X, C, A, rho, eps, lamda, optim_modes, ot_mode = "fiber", C_full = None, U_init = None, unbal = True, **kwargs):
        super(CoreModel, self).__init__(**kwargs)
        self.unbal = unbal
        self.X_shape = X.shape
        self.X = [tl.unfold(X, i) for i in range(len(X.shape))] # all unfoldings
        self.X_orig = X # folded tensor
        self.alpha = [(self.X[i].sum(0) > 0)*1.0 for i in range(len(X.shape))]
        self.C = C
        if C is not None:
            self.K = [(-C[k]/eps[k]).exp() for k in range(len(C))] # Gibbs kernels for each mode
        if C_full is not None:
            self.C_full = C_full
            self.K_full = (-C_full/eps[0]).exp()
        self.eps = eps
        self.rho = rho
        self.lamda = lamda
        self.A = A
        self.optim_modes = optim_modes
        self.ot_mode = ot_mode
        if U_init is None:
            U = [torch.zeros(self.X[i].shape).to(self.device) for i in range(len(X.shape))]
        else:
            U = U_init
        self.set_U(U)
    def get_omega(self, i):
        return tl.tenalg.multi_mode_dot(tl.fold(self.U[i], mode = i, shape = self.X_shape), [A.T for A in self.A])
    def dual_obj(self):
        if self.ot_mode == "fiber":
            return torch.sum(torch.stack([self.H_star_unbal(self.U[i], self.X[i], self.eps[i], self.lamda[i], self.K[i], alpha = self.alpha[i]) for i in self.optim_modes])) + \
                  self.rho[-1]*self.E_star_unbal((-1/self.rho[-1]) * sumstack([self.get_omega(i) for i in self.optim_modes]))
        elif self.ot_mode == "slice":
            return self.H_star_unbal(self.U[0].T, self.X[0].T, self.eps[0], self.lamda[0], self.K_full) + \
                  self.rho[-1]*self.E_star_bal((-1/self.rho[-1]) * sumstack([self.get_omega(i) for i in self.optim_modes]))
    def forward(self):
        return self.dual_obj()
    def compute_primal_variable(self):
        return torch.exp((-1/self.rho[-1])*sumstack([self.get_omega(i) for i in self.optim_modes]))
    def primal_obj(self, terms = False):
        return 0

def solve(model, lr = 1, tol = 0.01, max_iter = 100, print_inter = 10, check_iter = 10, mode = "lbfgs", retries = 4, factor = 2):
    def get_optimizer(model):
        if mode == "lbfgs":
            optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, history_size = 25, max_iter = 25, line_search_fn="strong_wolfe")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        return optimizer
    optimizer = get_optimizer(model)
    U_prev = [u.clone() for u in model.U]
    dual_prev = None
    for i in range(max_iter):
        got_nan = False
        def closure():
            optimizer.zero_grad()
            obj = model.dual_obj()
            obj.backward()
            if model.ot_mode == "fiber":
                for k in range(len(model.U)):
                    if model.U[k].grad is not None:
                        model.U[k].grad[:, model.alpha[k] == 0] = 0
            return obj
        optimizer.step(closure = closure)
        if i % check_iter == 0:
            with torch.no_grad():
                d = model.dual_obj().item()
                if dual_prev is not None and np.abs(d - dual_prev) < tol:
                    break
                else:
                    dual_prev = d
            # retries
            if math.isnan(d):
                got_nan = True
                if retries > 0:
                    lr = lr/factor
                    retries -= 1
                    # reset model to last stored variable state
                    print("Warning: NaN encountered, new lr = %f" % lr)
                    model.set_U(U_prev)
                    optimizer = get_optimizer(model)
                else:
                    print("Error: exhausted retries")
                    break # quit 
            else:
                # retain dual variables
                U_prev = [u.clone() for u in model.U]
        if i % print_inter == 0 and ~got_nan:
            print("i = %d \t dual = %f" % (i, d))
