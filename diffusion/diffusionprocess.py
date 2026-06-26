import torch
class DiffusionProcess:
    def __init__(self, schedule, parameterization, weight_func, time_param,  t_distribution, network):
        self.time_param = time_param
        self.schedule =schedule
        self.weight_func = weight_func
        self.parameterization = parameterization
        self.t_distribution = t_distribution
        self.network = network
    
    def forward_sample(self, x_0, t, noise, cond=None):
        '''
        Use the schedule to generate a noised sample.
        x_t = alpha_t x_0 + sigma_t epsilon
        '''
        return  self.schedule.alpha(t)*x_0 + self.schedule.sigma(t)*noise
    
    def training_loss(self, x_0, noise=None, cond= None):
        #u is the defined time param, there is a bijection between u and t
        #if the time param is Identity, t=u
        t, u = self.t_distribution.sample(sample_shape=(x_0.shape[0],1))
        #noise is sampled fresh unless curated noise is supplied (e.g. reflow pairs)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.forward_sample(x_0, t, noise) #back to t
        #parameterization: defined in t
        target = self.parameterization.training_target(x_0, noise, t)
        #weight: defined in t (existing weight funcs are written in t)
        weight = self.weight_func(t)
        #network: defined in u
        pred = self.network(x_t, u, cond=cond)
        loss = (weight*(pred-target).pow(2)).mean()
        clean_loss = None
        if hasattr(self.parameterization, "to_x_0"):
            clean_loss = (weight*(x_0 - self.parameterization.to_x_0(pred, x_t, t)).pow(2)).mean()
        return loss, clean_loss
    
    
    def predict_noise(self, x_t, t, cond=None):
        u = self.time_param.t_to_u(t)
        return self.parameterization.to_noise(self.network(x_t, u, cond=cond), x_t, t)
    
    def predict_x_0(self, x_t, t, cond=None):
        u = self.time_param.t_to_u(t)
        return self.parameterization.to_x_0(self.network(x_t, u, cond=cond), x_t, t)
    
    def predict_score(self, x_t, t, cond=None):
        u = self.time_param.t_to_u(t)
        return self.parameterization.to_score(self.network(x_t, u, cond=cond), x_t, t)
    
    def predict_v(self, x_t, t, cond=None):
        u = self.time_param.t_to_u(t)
        return self.parameterization.to_v(self.network(x_t, u, cond=cond), x_t, t)
    

        