"""
DEIS implementation by the book (p 282). The weights C_{i,j} in front of the buffer of previous noise
predictions are computed numerically with a gauss-legendre method.

For the weights: the integrating factor is simple when working in alpha and sigma.
Let E be the integrating factor:
E(s->t) = exp(int_s^t f(u) du) = alpha_t/alpha_s
g^2 is also easy known in terms of the canonical alpha_t and sigma_t
g^2 = 2sigma_t*sigma_t_prime - 2*(alpha_t_prime/alpha_t)*sigma_t^2
"""

import torch
import torch.nn as nn
import numpy as np


class DEISSolver:
    def __init__(self, model, parameterization, time_param, N, M=3, schedule=None):
        self.model = model
        self.time_param = time_param
        self.steps = N
        self.M = M
        self.parameterization = parameterization
        self.schedule = schedule

    def integrand_on_grid(self, tau, ts, i, j, m):
        """
        Vectorized function that accepts a vector integration grid used to compute the integrand for the weights C_{i,j} along the grid
        tau: (K,).
        The code below is very commented because the vectorization can be hard to follow.
        """
        # make this vectorized so tau can be a vector of tau used to approx the integral that defines the weights
        ts = np.asarray(ts, float)
        tau = np.asarray(tau, float)
        t_i_1_k = ts[i - 1 - np.arange(m)]
        t_i_1_j = ts[i - 1 - j]
        # for every tau we define m which are all the terms in the product of taus and ts that defines the lagrange polynomial
        # we have to do some broadcasting tricks
        num = tau[:, None] - t_i_1_k[None, :]  # (K,1) - (1, m) which gives (K,m)
        den = t_i_1_j - t_i_1_k  # (m,)
        den[j] = 1.0  # avoid dividing by 0 when t-i-k = t-i-j
        ratios = num / den[None, :]  # (K,m)
        ratios[:, j] = 1.0  # remove j from the sum
        prod = np.prod(ratios, axis=1)
        # integrating factor
        alpha_t_i = self.schedule.alpha(ts[i])
        alpha_tau = self.schedule.alpha(tau)
        alpha_prime_tau = self.schedule.alpha_prime(tau)
        sigma_prime_tau = self.schedule.sigma_prime(tau)
        sigma_tau = self.schedule.sigma(tau)
        E = alpha_t_i / alpha_tau
        g2_div_sigma = (
            2 * sigma_prime_tau - 2 * (alpha_prime_tau / alpha_tau) * sigma_tau
        )  # already divided out the sigma_tau
        return 0.5 * (g2_div_sigma) * E * prod

    def deis_coeff(
        self, ts, i, j, m, n_quad=64
    ):  # might just hardcode 64, the weights aren't hard to approximate
        # integral computed with gauss legendre method
        a, b = ts[i - 1], ts[i]
        x, wq = np.polynomial.legendre.leggauss(n_quad)
        tau = 0.5 * (b - a) * x + 0.5 * (a + b)
        return 0.5 * (b - a) * (wq * self.integrand_on_grid(tau, ts, i, j, m)).sum()

    def precompute_weights(self, time_grid, n=None, n_quad=64):
        # hardcoded n_quad because this should always be fine to compute the weights accurately
        if n is None:
            n = self.K
        ts = np.asarray(time_grid, float)
        N = len(ts) - 1
        C, E = [], np.empty(N)
        for i in range(1, N + 1):
            m = min(n, i - 1) + 1
            C.append(np.array([self.deis_coeff(ts, i, j, m, n_quad) for j in range(m)]))
            E[i - 1] = float(
                self.schedule.alpha(ts[i]) / self.schedule.alpha(ts[i - 1])
            )
        self.ts, self.C, self.E = ts, C, E
        return C, E

    def model_eval(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)

    def sample(self, x0, return_trajectory=False, eps=None, time_grid=None):
        device = x0.device
        ts = self.time_param.grid(self.steps, eps=eps, device=device)
        if time_grid is not None:
            ts = time_grid
        ts_np = (
            ts.detach().cpu().numpy().astype(float)
        )  # need this to be numpy for the weight computation
        self.precompute_weights(ts_np, n=self.M)

        x = x0.clone()
        traj = [x.clone()] if return_trajectory else None
        eps_buf = []

        for i in range(self.steps):
            t = ts[i]
            u = self.time_param.t_to_u(t)
            u_tens = torch.full((x.shape[0], 1), u.item(), device=device, dtype=x.dtype)

            model_out = self.model_eval(x, u_tens)
            eps = self.parameterization.to_noise(model_out, x, t)
            eps_buf.insert(0, eps)
            eps_buf = eps_buf[: self.M + 1]

            Ci = self.C[i]
            x = float(self.E[i]) * x
            for j in range(len(Ci)):
                x = x + float(Ci[j]) * eps_buf[j]

            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return x, torch.stack(traj)
        return x
