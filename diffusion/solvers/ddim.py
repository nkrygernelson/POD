import torch

class DDIMSolver:

    def __init__(self, model, parameterization,time_param, schedule, N):
        self.model = model
        self.pmtrz = parameterization
        self.time_param = time_param
        self.schedule = schedule
        self.steps = N
    def model_eval(self, x, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t)

    def sample(self, x0, eps = 0, N=None, return_trajectory=False):
        '''
        eps: value to give as buffer around the ends of the grid
        '''
        if not N:
            N = self.steps

        device=x0.device
        ts = self.time_param.grid(N, device=device, eps=eps)
        x = x0.clone()
        traj = [x.clone()] if return_trajectory else None

        for i in range(N):
            s = ts[i].item()       
            t = ts[i + 1].item()   
            s_tens = torch.full((x.shape[0], 1), s, device=device)
            u_tens = self.time_param.t_to_u(s_tens)
            model_out = self.model_eval(x, u_tens)
            x_star = self.pmtrz.to_x_0(model_out, x, s_tens)
            eps_star = self.pmtrz.to_noise(model_out, x, s_tens)
            alpha_t = self.schedule.alpha(t)
            sigma_t = self.schedule.sigma(t)
            x = alpha_t*x_star + sigma_t*eps_star
            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return x, torch.stack(traj)
        return x