import torch
class Uniform_u:
    def __init__(self,device, time_param, eps = 1e-4, u = 1.0, l =0.0):
        self.time_param = time_param
        lo, hi = sorted((time_param.u_min, time_param.u_max))
        self.t_distribution = torch.distributions.Uniform(
        torch.tensor(lo, device=device),
        torch.tensor(hi, device=device),)
    def sample(self,sample_shape):
        u = self.t_distribution.sample(sample_shape=sample_shape)
        t = self.time_param.u_to_t(u)
        return t, u
    
class Uniform_t:
    def __init__(self,device, time_param, eps = 1e-4, u = 1.0, l =0.0):
        self.time_param = time_param
        self.t_distribution = torch.distributions.Uniform(
        torch.tensor(l+eps, device=device),
        torch.tensor(u-eps, device=device),)
    def sample(self,sample_shape):
        t = self.t_distribution.sample(sample_shape=sample_shape)
        u = self.time_param.t_to_u(t)
        return t, u
