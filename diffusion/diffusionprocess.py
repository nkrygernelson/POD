import torch
class DiffusionProcess:
    def __init__(self, schedule, parameterization, weight_func, t_distribution, network):
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
    
    def training_loss(self, x_0, cond= None):
        t = self.t_distribution.rsample(sample_shape=(x_0.shape[0],1))
        noise = torch.randn_like(x_0)
        x_t = self.forward_sample(x_0, t, noise)
        target = self.parameterization.training_target(x_0, noise, t)
        weight = self.weight_func(t)
        pred = self.network(x_t, t, cond=cond)
        loss = (weight*(pred-target).pow(2)).mean()
        clean_loss = None
        if hasattr(self.parameterization, "to_x_0"):
            clean_loss = (weight*(x_0 - self.parameterization.to_x_0(pred, x_t, t)).pow(2)).mean()
        return loss, clean_loss
    
    def clean_loss(self, x_0, cond = None):
        t = self.t_distribution.rsample(sample_shape=(x_0.shape[0],1))
        noise = torch.randn_like(x_0)
        x_t = self.forward_sample(x_0, t, noise)
        clean = self.parameterization.to_clean(x_0)
    
    def predict_noise(self, x_t, t, cond=None):
        return self.parameterization.to_noise(self.network(x_t, t, cond=cond), x_t, t)
    
    def predict_x_0(self, x_t, t, cond=None):
        return self.parameterization.to_x_0(self.network(x_t, t, cond=cond), x_t, t)
    
    def predict_score(self, x_t, t, cond=None):
        return self.parameterization.to_score(self.network(x_t, t, cond=cond), x_t, t)
    

        