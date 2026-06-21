import torch

class v_PFode:
    def __init__(self, schedule):
        self.schedule = schedule

    def diff_xt(self,xt, t, model_out):
        return model_out

class clean_PFode:
    def __init__(self, schedule):
        self.schedule = schedule

    def diff_xt(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (sp/s)*xt + a*(ap/a -sp/s)*model_out
    
class noise_PFode:
    def __init__(self, schedule):
        self.schedule = schedule

    def diff_xt(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (ap/a)*xt - s*(ap/a -sp/s)*model_out
    
class score_PFode:
    def __init__(self, schedule):
        self.schedule = schedule

    def diff_xt(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (ap/a)*xt - (s**2)*(ap/a -sp/s)*model_out


    

    
