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

def sumstack(l):
    if len(l) > 0:
        return torch.stack(l).sum(0)
    else:
        return 0


normalise = lambda x: x/x.sum()

class OTModel(torch.nn.Module):
    '''
    Base variational OT problem class. 
    '''
    def __init__(self, device = None):
        super(OTModel, self).__init__()
        self.device = torch.device("cpu") if device is None else device
        pass
    def E(self, p):
        '''
        Entropy on positive measures, E(p) = <p, log(p)-1>
        '''
        return (xlogy(p, p) - p).sum()
    def E_star_simplex(self, p):
        '''
        Legendre transform of E(p) subject to simplex constraint <p, 1> = 1 along axis 0.
        '''
        return torch.logsumexp(p, 0).sum()
    def E_star(self, p):
        '''
        Legendre transform of E(p) on positive measures.
        '''
        return p.exp().sum()
    def genKL(self, a, b):
        '''
        Generalised KL-divergence on positive measures, KL(a, b) = <a, log(a/b)> - <a, 1> + <b, 1>
        '''
        return (xlogy(a, a/b) - a + b).sum()
    def KL(self, a, b):
        '''
        KL-divergence on positive measures, KL(a, b) = <a, log(a/b)-1>
        '''
        return (xlogy(a, a/b) - a).sum()
    def OT_semidual_bal(self, U, P, eps, K):
        '''
        Dual of OT(P, Q) in its second argument, i.e. dual pairing is <Q, U>. 
        Handles the case of batch OT, where columns of P are histograms on a common space.
        '''
        X = U/eps
        scale = X.max(0).values
        return eps*(-self.E(P) + torch.sum(P*(scale + torch.log(K @ torch.exp(X-scale)))))
    def OT_semidual_bal_tensor(self, U, P, eps, K):
        '''
        Dual of OT(P, Q) in its second argument, i.e. dual pairing is <Q, U>. 
        Handles the case of tensor OT, where P, U are d-way tensors, and K is a list of d Gibbs kernels. 
        '''
        X = U/eps
        scale = X.max()
        return eps*(-self.E(P) + torch.sum(P*(scale + torch.log(tl.tenalg.multi_mode_dot(torch.exp(X-scale), K)))))
    def get_ufrac(self, u, lamda, eps, log = False):
        '''
        Compute f = (lambda/(lambda-u))^(lambda/epsilon) for unbalanced OT. 
        If log == True, result is returned in log-domain.
        '''
        if log:
            return (lamda/eps)*torch.log(lamda/(lamda - u))
        else:
            return ((lamda/eps)*torch.log(lamda/(lamda - u))).exp()
    def OT_semidual_unbal_tensor(self, U, P, eps, lamda, K):
        '''
        Dual of OTU(P, Q) in its second argument, i.e. dual pairing is <Q, U>. 
        Handles the case of batch OT, where columns of P are histograms on a common space.
        '''
        F = self.get_ufrac(U, lamda, eps, log = True)
        scale = F.max()
        return -eps*xlogy(P, P).sum() + eps*xlogy(P, tl.tenalg.multi_mode_dot((F - scale).exp(), K)).sum() + eps*(P*(scale + 1)).sum()
    def OT_semidual_unbal(self, U, P, eps, lamda, K, alpha = None):
        '''
        Dual of OTU(P, Q) in its second argument, i.e. dual pairing is <Q, U>. 
        Handles the case of tensor OT, where P, U are d-way tensors, and K is a list of d Gibbs kernels. 
        '''
        F = self.get_ufrac(U, lamda, eps, log = True)
        scale = F.max(0).values
        if alpha is None:
            return -eps*xlogy(P, P).sum() + eps*xlogy(P, K @ (F - scale).exp()).sum() + eps*(P*(scale + 1)).sum()
        else:
            return -eps*xlogy(P, P).sum(0).dot(alpha) + eps*xlogy(P, K @ (F - scale).exp()).sum(0).dot(alpha) + eps*(P*(scale + 1)).sum(0).dot(alpha)
    # Commented out for now because this is extremely slow
    # def OT_semidual_unbal_stab(self, U, P, eps, lamda, C):
    #     F = self.get_ufrac(U, lamda, eps, log = True)
    #     return -eps*xlogy(P, P).sum() + eps*P.sum() + \
    #             eps*xy(P, torch.logsumexp((-C/eps).reshape(C.shape[0], C.shape[0], 1) + F.reshape(1, F.shape[0], F.shape[1]), dim = 1)).sum()
    def OT_unbal(self, U, P, eps, lamda, K):
        '''
        Compute primal OTU(P, Q) from dual potential U at optimality. 
        Aggregates along columns.
        '''
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

class FactorsModel(OTModel):
    '''
    Subclass for factor matrix subproblem
    
    X: data tensor
    k: index of factor matrix we want 
    C: list of cost matrices (one for each mode)
    S: core tensor (can be constant)
    A: list of factor matrices
    rho: factor matrix entropy weights
    eps: OT regularisation parameter
    lamda: unbalanced OT parameter (only relevant if unbal = True)
    ot_mode: "fiber", "slice", or "full"
    U_init: initial dual potential to use (if None, U initialised as 0)
    unbal: whether to use unbalanced transport
    norm: None, "row", "col" or "full". How to (simplex)-normalise the factor matrix.
    '''
    def __init__(self, X, k, C, S, A, rho, eps, lamda, ot_mode = "slice", U_init = None, unbal = True, norm = None, **kwargs):
        super(FactorsModel, self).__init__(**kwargs)
        self.X = X # folded tensor
        self.k = k 
        self.C = C
        self.unbal = unbal
        self.norm = norm
        self.K = [(-C[k]/eps[k]).exp() for k in range(len(C))] # Gibbs kernels for each mode
        self.eps = eps
        self.rho = rho
        self.lamda = lamda
        self.A = A
        if U_init is None:
            U = torch.zeros(self.X.shape).to(self.device)
        else:
            U = U_init
        self.S = S
        self.set_U(U)
        self.ot_mode = ot_mode
    def get_xi(self, i):
        modes_a = np.arange(i+1, len(self.A))[::-1]
        modes_b = np.arange(0, i)
        A=tl.unfold(tl.tenalg.multi_mode_dot(self.U, [self.A[j].T for j in modes_a], modes = modes_a), i)
        B=tl.unfold(tl.tenalg.multi_mode_dot(self.S, [self.A[j] for j in modes_b], modes = modes_b), i).T
        return A @ B
    def set_U(self, U):
        self.U = torch.nn.Parameter(Variable(U, requires_grad = True))
    def dual_obj(self):
        '''
        Compute dual objective
        '''
        Z = (-1/self.rho[self.k]) * self.get_xi(self.k)
        if self.norm is None:
            ent_term = self.rho[self.k]*self.E_star(Z)
        elif self.norm == "col":
            ent_term = self.rho[self.k]*self.E_star_simplex(Z)
        elif self.norm == "row":
            ent_term = self.rho[self.k]*self.E_star_simplex(Z.T)
        elif self.norm == "full":
            ent_term = self.rho[self.k]*self.E_star_simplex(Z.reshape(-1))
        # case: OT along fibers. need to use the alpha mask to drop zero fibers.
        # case: OT along slices. we need the rows of X_(0) to be the slices, and use C[0] as the cost. 
        if self.ot_mode == "slice":
            ot_term = self.OT_semidual_unbal(tl.unfold(self.U, 0).T, tl.unfold(self.X, 0).T, self.eps[0], self.lamda[0], self.K[0]) \
                    if self.unbal else self.OT_semidual_bal(tl.unfold(self.U, 0).T, tl.unfold(self.X, 0).T, self.eps[0], self.K[0])
        elif self.ot_mode == "full":
            ot_term = self.OT_semidual_unbal_tensor(self.U, self.X, self.eps[0], self.lamda[0], self.K) if self.unbal else \
                        self.OT_semidual_bal_tensor(self.U, self.X, self.eps[0], self.K)
        return ot_term + ent_term
    def primal_obj(self, terms = False):
        # not implemented 
        return 0
    def forward(self):
        return self.dual_obj()
    def compute_primal_variable(self):
        '''
        Get factor matrix (the primal variable) at optimality.
        '''
        Z = (-1/self.rho[self.k]) * self.get_xi(self.k)
        a = torch.exp(Z)
        if self.norm is None:
            return a 
        elif self.norm == "col":
            return a/a.sum(0).reshape(1, -1)
        elif self.norm == "row":
            return (a.T/a.sum(1)).T
        elif self.norm == "full":
            return a/a.sum()
#
class CoreModel(OTModel):
    '''
    Subclass for core tensor subproblem
    
    X: data tensor
    C: list of cost matrices (one for each mode)
    A: list of factor matrices
    rho: factor matrix entropy weights
    eps: OT regularisation parameter
    lamda: unbalanced OT parameter (only relevant if unbal = True)
    optim_modes: modes along which to compute OT
    ot_mode: "fiber", "slice", or "full"
    U_init: initial dual potential to use (if None, U initialised as 0)
    unbal: whether to use unbalanced transport
    norm: None, "row", "col" or "full". How to (simplex)-normalise the factor matrix.
    '''
    def __init__(self, X, C, A, rho, eps, lamda, ot_mode = "slice", U_init = None, unbal = True, norm = None, **kwargs):
        super(CoreModel, self).__init__(**kwargs)
        self.X = X # folded tensor
        self.C = C
        self.K = [(-C[k]/eps[k]).exp() for k in range(len(C))] # Gibbs kernels for each mode
        self.unbal = unbal
        self.eps = eps
        self.rho = rho
        self.lamda = lamda
        self.A = A
        self.ot_mode = ot_mode
        self.norm = norm
        if U_init is None:
            U = torch.zeros(self.X.shape).to(self.device)
        else:
            U = U_init
        self.set_U(U)
    def set_U(self, U):
        self.U = torch.nn.Parameter(Variable(U, requires_grad = True))
    def get_omega(self):
        return tl.tenalg.multi_mode_dot(self.U, [A.T for A in self.A])
    def dual_obj(self):
        Z = (-1/self.rho[-1]) * self.get_omega()
        ot_term = None
        ent_term = None
        if self.norm is None:
            ent_term = self.rho[-1]*self.E_star(Z)
        elif self.norm == "full":
            ent_term = self.rho[-1]*self.E_star_simplex(Z.reshape(-1))
        if self.ot_mode == "slice":
            ot_term = self.OT_semidual_unbal(tl.unfold(self.U, 0).T, tl.unfold(self.X, 0).T, self.eps[0], self.lamda[0], self.K[0]) \
                    if self.unbal else self.OT_semidual_bal(tl.unfold(self.U, 0).T, tl.unfold(self.X, 0).T, self.eps[0], self.K[0])
        elif self.ot_mode == "full":
            ot_term = self.OT_semidual_unbal_tensor(self.U, self.X, self.eps[0], self.lamda[0], self.K) if self.unbal else \
                        self.OT_semidual_bal_tensor(self.U, self.X, self.eps[0], self.K)
        return ot_term + ent_term
    def forward(self):
        return self.dual_obj()
    def compute_primal_variable(self):
        s = torch.exp((-1/self.rho[-1])*self.get_omega())
        if self.norm is None:
            return s
        elif self.norm == "full":
            return s/s.sum()
    def primal_obj(self, terms = False):
        return 0

def solve(model, lr = 1, tol = 0.01, max_iter = 100, print_inter = 10, check_iter = 10, mode = "lbfgs", retries = 4, factor = 2):
    def get_optimizer(model):
        if mode == "lbfgs":
            optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, history_size = 10, max_iter = 10, line_search_fn="strong_wolfe")
        elif mode == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        elif mode == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        return optimizer
    optimizer = get_optimizer(model)
    U_prev = model.U.clone()
    dual_prev = None
    for i in range(max_iter):
        got_nan = False
        def closure():
            optimizer.zero_grad()
            obj = model.dual_obj()
            obj.backward()
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
                U_prev = model.U.clone()
        if i % print_inter == 0 and ~got_nan:
            print("i = %d \t dual = %f" % (i, d))
    return dual_prev