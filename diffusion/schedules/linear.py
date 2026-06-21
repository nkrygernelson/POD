
'''
linear schedule. We assume that the starting time is zero (clean) and the ending time is 1 (noised)
'''
class Linear:
    
    @staticmethod
    def alpha(t):
        tol = 1e-5
        return 1-t + tol
    @staticmethod
    def sigma(t):
        return t + 1e-5
    @staticmethod
    def alpha_prime(t):
        return -1
    @staticmethod
    def sigma_prime(t):
        return 1
    
