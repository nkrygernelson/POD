
import torch
import torch.nn as nn
class DEISSolver:
    def __init__(self , model, parameterization, N, ti = 1, tf=0, schedule=None):
        self.model = model
        self.steps = N
        self.ti = ti
        self.tf = tf
        self.parameterziation = parameterization
    def LP(ys, m, t):
        l = []
        t_tens = t*torch.ones(size=(m,1))


                

    def model_eval(self, x,t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x,t)

    
         
    def sample(self, x0, return_trajectory =False):
        N_steps = self.steps
        step_size = (self.tf-self.ti)/N_steps
        x = x0.clone()
        traj = [x.clone()] if return_trajectory else None
        for i in range(N_steps):
            t = self.ti +i*step_size
            t_tens = torch.full((x.shape[0],1), t)
            model_out = self.model_eval(x, t_tens)
            x = x + step_size*self.parameterziation.pfode(x, t, model_out)
            if return_trajectory:
                traj.append(x.clone())
        if return_trajectory:
            return x, torch.stack(traj)
        return x

