import torch

#define a linear scheule as a strating point
class DiffusionScheduler:
    '''
    This is the mega class that given a sigma_t and alpha_t functions we can create everything that we would need to train in given formalism.
    This class returns the equivalent learning objectives for every parameterization.
    '''
    def __init__(self,alpha_t, sigma_t, t_weigh_func = 1,):
        '''
        alpha_t: func
        sigma_t: func
        these are terms in front of the affine flow: alpha_t x_0 + sigma_t epsilon
        '''
        self.alpha = alpha_t
        self.sigma = sigma_t

    
    