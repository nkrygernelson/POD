'''
Time class
'''
import torch

class TimeParam:
    '''
    Template structure for the time object.
    Maps between schedule time t and a working coordinate u.
    Subclasses must set u_min/u_max and  u_to_t / t_to_u.
    '''
    u_min = None
    u_max = None

    def __init__(self, schedule):
        self.schedule = schedule
    def u_to_t(self, u):
        raise NotImplementedError
    def t_to_u(self, t):
        raise NotImplementedError
    def grid(self, N, device="cpu", eps = 0):
        u = torch.linspace(self.u_min+eps, self.u_max-eps, N + 1, device=device)
        return self.u_to_t(u)
    
class Identity(TimeParam):
    def __init__(self, schedule=None):
        self.u_min, self.u_max = 1.0 - 1e-3, 0.0 + 1e-3
    def t_to_u(self, t): return t
    def u_to_t(self, u): return u

class LogSNRTime(TimeParam):
    def __init__(self, schedule, u_min=-10.0, u_max=10.0):
        super().__init__(schedule)
        self.u_min, self.u_max = u_min, u_max 
    def t_to_u(self, t):
        return 2 * (torch.log(self.schedule.alpha(t)) - torch.log(self.schedule.sigma(t)))
    def u_to_t(self, u):
        return torch.sigmoid(-u/2)