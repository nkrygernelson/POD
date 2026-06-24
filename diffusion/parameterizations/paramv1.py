import torch

class ScorePred:
    #check
    def __init__(self, schedule):
        self.schedule = schedule
        self.ptype = "score_pred"
    def training_target(self, x_0, noise, t):
        return -noise/self.schedule.sigma(t)
    def to_noise(self, model_out, x_t, t): 
        return model_out*(-self.schedule.sigma(t))
    def to_x_0  (self, model_out, x_t, t): 
        return (1/self.schedule.alpha(t))*x_t + (self.schedule.sigma(t).pow(2)/self.schedule.alpha(t))*model_out
    def to_score(self, model_out, x_t, t): 
        return model_out 
    def to_v(self, model_out, x_t, t):
        ap = self.schedule.alpha_prime(t)
        sp = self.schedule.sigma_prime(t)
        x_0 = self.to_x_0(model_out, x_t, t)
        noise = self.to_noise(model_out, x_t, t)
        return ap*x_0 + sp*noise
    
    def pfode(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (ap/a)*xt + (s**2)*(ap/a -sp/s)*model_out


class VPred:
    '''
    going from alpha/sigma to f(t) and g^2(t) is tricky
    '''
    def __init__(self, schedule):
        self.schedule = schedule
        self.ptype = "v_pred"

    def training_target(self, x_0, noise, t):
        return self.schedule.alpha_prime(t)*x_0+self.schedule.sigma_prime(t)*noise
    
    def to_v(self, model_out, x_t, t):
        return model_out
    
    def pfode(self,xt, t, model_out):
        return model_out
    

class NoisePred:
    #check
    def __init__(self, schedule):
        self.schedule = schedule
        self.ptype = "noise_pred"
    def training_target(self, x_0, noise, t):
        return noise
    def to_noise(self, model_out, x_t, t): 
        return model_out
    def to_x_0  (self, model_out, x_t, t): 
        return (x_t - self.schedule.sigma(t) * model_out) / self.schedule.alpha(t)
    def to_score(self, model_out, x_t, t): 
        return -model_out / self.schedule.sigma(t)
    def to_v(self, model_out, x_t, t):
        ap = self.schedule.alpha_prime(t)
        sp = self.schedule.sigma_prime(t)
        x_0 = self.to_x_0(model_out, x_t, t)
        noise = self.to_noise(model_out, x_t, t)
        return ap*x_0 + sp*noise
    
    def pfode(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (ap/a)*xt - s*(ap/a -sp/s)*model_out
    
class CleanPred:
    def __init__(self, schedule):
        self.schedule = schedule
        self.ptype = "clean_pred"
    
    def training_target(self, x_0, noise, t):
        return x_0
    
    def to_noise(self, model_out, x_t, t):
        alpha_t = self.schedule.alpha(t)
        sigma_t = self.schedule.sigma(t) 
        return (model_out- (1/alpha_t)*x_t)*(-alpha_t/sigma_t)
    def to_x_0  (self, model_out, x_t, t): 
        return model_out
    def to_score(self, model_out, x_t, t):
        #(x* -frac{1}{\alpha_t}x_t)\frac{alpha_t}{sigma**2} 
        alpha_t = self.schedule.alpha(t)
        sigma_t = self.schedule.sigma(t)
        return (model_out- (1/alpha_t)*x_t)*(alpha_t/sigma_t.pow(2))
    def to_v(self, model_out, x_t, t):
        ap = self.schedule.alpha_prime(t)
        sp = self.schedule.sigma_prime(t)
        x_0 = self.to_x_0(model_out, x_t, t)
        noise = self.to_noise(model_out, x_t, t)
        return ap*x_0 + sp*noise
    def pfode(self, xt, t, model_out):
        a = self.schedule.alpha(t)
        ap = self.schedule.alpha_prime(t)
        s = self.schedule.sigma(t)
        sp = self.schedule.sigma_prime(t)
        return (sp/s)*xt + a*(ap/a -sp/s)*model_out
    
    