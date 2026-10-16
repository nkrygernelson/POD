"""
This implmentation of ddim use the last equation of 9.2.3 as the DDIM update rule:
For s > t
hat{x_t} = alpha_t * x_pred(x_s, s) + sigma_t * noise_pred(x_s, s)

Numerically the most efficient approach would be to apply the DDIM update rule (
from the other equations in 9.23) that corresponds to the parameterization of the trained model. 
Here, we convert our model output to the x_0 and noise predictions by calling to_noise or to_x_0.
Such conversions are stable so long as the determinant of the transformation (see paramv2) is not 
0 or near 0: this can be checked in the "singularity_plot" produced during training.

"""

import torch


class DDIMSolver:

    def __init__(self, model, parameterization, time_param, schedule, N):
        self.model = model
        self.pmtrz = parameterization
        self.time_param = time_param
        self.schedule = schedule
        self.steps = N

    def model_eval(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)

    def sample(self, x0, eps=0, N=None, return_trajectory=False, time_grid=None):
        """
        eps: value to give as buffer around the ends of the grid
        """
        if not N:
            N = self.steps

        device = x0.device
        ts = self.time_param.grid(N, device=device, eps=eps)
        if time_grid is not None:  # override time grid
            ts = time_grid
        x = x0.clone()
        traj = [x.clone()] if return_trajectory else None

        for i in range(N):
            s = ts[i].item()
            t = ts[i + 1].item()
            s_tens = torch.full((x.shape[0],) + (1,)*(x.dim()-1), s, device=device)
            u_tens = self.time_param.t_to_u(s_tens)
            model_out = self.model_eval(x, u_tens)
            x_star = self.pmtrz.to_x_0(model_out, x, s_tens)
            eps_star = self.pmtrz.to_noise(model_out, x, s_tens)
            alpha_t = self.schedule.alpha(t)
            sigma_t = self.schedule.sigma(t)
            x = alpha_t * x_star + sigma_t * eps_star
            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return x, torch.stack(traj)
        return x
