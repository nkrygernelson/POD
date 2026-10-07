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

    def __init__(self, schedule, eps = 0):
        self.schedule = schedule
        self.eps = eps
    def u_to_t(self, u):
        raise NotImplementedError
    def t_to_u(self, t):
        raise NotImplementedError
    def u_bounds(self, eps=None):
        if eps is None:
            eps = self.eps
        step = eps if self.u_max >= self.u_min else -eps
        return self.u_min + step, self.u_max - step
    def grid(self, N, device="cpu", eps = None):
        lo, hi = self.u_bounds(eps)
        u = torch.linspace(lo, hi, N + 1, device=device)
        return self.u_to_t(u)
    
class Identity(TimeParam):
    def __init__(self, schedule=None, eps=0):
        super().__init__(schedule, eps)
        self.u_min, self.u_max = 1.0 , 0.0
    def t_to_u(self, t): return t
    def u_to_t(self, u): return u

class LogSNRTime(TimeParam):
    def __init__(self, schedule, u_min=-10.0, u_max=10.0, eps=0):
        super().__init__(schedule, eps)
        self.u_min, self.u_max = u_min, u_max
    def t_to_u(self, t):
        return 2 * (torch.log(self.schedule.alpha(t)) - torch.log(self.schedule.sigma(t)))
    def u_to_t(self, u):
        return torch.sigmoid(-u/2)