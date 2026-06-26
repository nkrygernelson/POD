import torch


class Uniform_u:
    '''Uniform in the working coordinate u, over the time_param's buffered u-bounds
    (the same support grid() covers). Returns (t, u).'''
    def __init__(self, device, time_param, eps=None):
        self.time_param = time_param
        lo, hi = sorted(time_param.u_bounds(eps))
        self.t_distribution = torch.distributions.Uniform(
            torch.tensor(lo, device=device),
            torch.tensor(hi, device=device),)

    def sample(self, sample_shape):
        u = self.t_distribution.sample(sample_shape=sample_shape)
        t = self.time_param.u_to_t(u)
        return t, u


class Uniform_t:
    '''Uniform in physical time t, over the t-image of the time_param's buffered
    u-bounds (so the support matches grid() / Uniform_u). Returns (t, u).'''
    def __init__(self, device, time_param, eps=None):
        self.time_param = time_param
        u_lo, u_hi = time_param.u_bounds(eps)
        t_a = float(time_param.u_to_t(torch.tensor(float(u_lo))))
        t_b = float(time_param.u_to_t(torch.tensor(float(u_hi))))
        lo, hi = sorted((t_a, t_b))
        self.t_distribution = torch.distributions.Uniform(
            torch.tensor(lo, device=device),
            torch.tensor(hi, device=device),)

    def sample(self, sample_shape):
        t = self.t_distribution.sample(sample_shape=sample_shape)
        u = self.time_param.t_to_u(t)
        return t, u
