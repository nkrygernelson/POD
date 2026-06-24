
'''
linear schedule. We assume that the starting time is zero (clean) and the ending time is 1 (noised)
'''
class Linear:
    
    @staticmethod
    def alpha(t):
        return 1-t +1e-5
    @staticmethod
    def sigma(t):
        return t +1e-5
    @staticmethod
    def alpha_prime(t):
        return -1
    @staticmethod
    def sigma_prime(t):
        return 1
    @staticmethod
    def gamma(t):
        #gamma t controls reverse SDE
        return 0.1
