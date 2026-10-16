import torch
import torch.nn as nn


class EulerMauryamaSolver:
    def __init__(self, model, parameterization, time_param, N, schedule=None):
        self.schedule = schedule
        self.time_param = time_param
        self.model = model
        self.steps = N
        self.parameterization = parameterization  

    def model_eval(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)

    def sample(self, x0, return_trajectory=False, eps=0, time_grid=None):
        gamma_func = self.schedule.gamma
        device =x0.device
        ts=self.time_param.grid(self.steps, eps = eps, device=device)
        if time_grid is not None:
            ts = time_grid
        x =x0.clone()
        traj = [x.clone()] if return_trajectory else None
        N = self.steps
        for i in range(N):
            t = ts[i]
            dt = ts[i+1]-ts[i]                             
            u = self.time_param.t_to_u(t)
            gamma = gamma_func(t)
            t_tens=torch.full((x.shape[0],) + (1,)*(x.dim()-1),t.item(),device=device)
            u_tens=torch.full((x.shape[0],1),u.item(),device=device)
            model_out=self.model_eval(x, u_tens)
            drift= self.parameterization.rev_sde(x, t_tens, model_out, gamma)
            #EM step with no noise on last step
            last = (i == N - 1)
            noise = 0.0 if last else gamma * torch.sqrt(torch.abs(dt)) * torch.randn_like(x)
            x = x + drift * dt + noise
            if return_trajectory:
                traj.append(x.clone())
        if return_trajectory:
            return x, torch.stack(traj)
        return x