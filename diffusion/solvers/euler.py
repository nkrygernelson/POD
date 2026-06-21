import torch
import torch.nn as nn
class EulerSolver:
    def __init__(self , model, pfode_func, N, ti = 1, tf=0):
        self.model = model
        self.steps = N
        self.ti = ti
        self.tf = tf
        self.pfode = pfode_func
    def model_eval(self, x,t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x,t) 
    def sample(self, x0):
        N_steps = self.steps
        step_size = (self.tf-self.ti)/N_steps
        x = x0.clone()
        for i in range(N_steps):
            t = self.ti +i*step_size
            t_tens = torch.full((x.shape[0], 1), t)
            model_out = self.model_eval(x, t_tens)
            #xt, t, model_out)
            x = x + step_size*self.pfode(x, t, model_out)
        return x

