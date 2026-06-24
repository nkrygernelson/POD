import torch
import torch.nn as nn


class EulerSolver:
    def __init__(self, model, parameterization, time_param, N, schedule=None):
        self.time_param = time_param
        self.model = model
        self.steps = N
        self.parameterization = parameterization  

    def model_eval(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)

    def sample(self, x0, return_trajectory=False):
        device =x0.device
        ts=self.time_param.grid(self.steps, device=device)
        x =x0.clone()
        traj = [x.clone()] if return_trajectory else None
        N = self.steps
        for i in range(N):
            t = ts[i]
            dt = ts[i+1]-ts[i]                             
            u = self.time_param.t_to_u(t)
            t_tens=torch.full((x.shape[0],1),t.item(),device=device)
            u_tens=torch.full((x.shape[0],1),u.item(),device=device)
            model_out=self.model_eval(x, u_tens)
            x = x+dt*self.parameterization.pfode(x,t,model_out)  
            if return_trajectory:
                traj.append(x.clone())
        if return_trajectory:
            return x, torch.stack(traj)
        return x