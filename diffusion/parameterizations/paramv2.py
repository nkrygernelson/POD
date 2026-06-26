"""
Parameterization base class, inhertied by specific parameterization subclasses.
The main idea of this class is the following transformation
[[xt         ] = [[alpha_t, sigma_t] . [[x_0]
[[model_out]]]   [[a      ,    b  ]]   [epsilon]]

where the coeffs a and b are determined by the parameterization
noise: a = 0, b = 1
x_0:   a = 1, b = 0
score: a = 0, b = (-1/sigma_t)
v:     a = alpha_t_', sigma_t'

We can define the inverse transformation by inverting the 2 by 2 matrix.

[[x_0         ] = 1/Det . [[b.     , -sigma_t] . [[x_0]
[[epsilon]]   ]           [[-a     , alpha_t]]   [epsilon]]

This transformation is only defined when the determinant is non_zero.
We can monitor the stability of the conversion by the monitoring the magnitude of the determinant

Why I wrote this V2 is because more can go wrong with a bunch of independent classes than
a single base class. Also, expanding this code is much easier than in v1 because we can add features to the base
class and don't need to independently update 4 indepepndent classes.
"""

import torch


class Parameterization:

    def __init__(self, schedule):
        self.schedule = schedule

    def coeffs(self, t):
        raise NotImplementedError

    def training_target(self, x_0, eps, t):
        a, b = self.coeffs(t)
        return a * x_0 + b * eps

    def to_x0_eps(self, model_out, x_t, t):  # the ONE inversion
        s = self.schedule
        a, b = self.coeffs(t)
        alpha, sigma = s.alpha(t), s.sigma(t)
        det = alpha * b - sigma * a
        x_0 = (b * x_t - sigma * model_out) / det
        eps = (-a * x_t + alpha * model_out) / det
        return x_0, eps

    def to_x_0(self, m, x, t):
        return self.to_x0_eps(m, x, t)[0]

    def to_noise(self, m, x, t):
        return self.to_x0_eps(m, x, t)[1]

    def to_score(self, m, x, t):
        return -self.to_noise(m, x, t) / self.schedule.sigma(t)

    def to_v(self, m, x, t):
        x0, eps = self.to_x0_eps(m, x, t)
        return self.schedule.alpha_prime(t) * x0 + self.schedule.sigma_prime(t) * eps

    def pfode(self, x, t, m):
        return self.to_v(m, x, t)

    def rev_sde(self, x, t, m, gamma):
        x0, eps = self.to_x0_eps(m, x, t)
        s = self.schedule
        v = s.alpha_prime(t) * x0 + s.sigma_prime(t) * eps
        score = -eps / s.sigma(t)
        return v - 0.5 * gamma**2 * score


# Not sure if this is necessary but we dont need to conversion to predict the qunatity the model already outputs
class CleanPred(Parameterization):
    def to_x_0(self, m, x, t):
        return m

    def coeffs(self, t):
        return 1.0, 0.0


class NoisePred(Parameterization):
    def to_noise(self, m, x, t):
        return m

    def coeffs(self, t):
        return 0.0, 1.0


class ScorePred(Parameterization):
    def to_score(self, m, x, t):
        return m

    def coeffs(self, t):
        return 0.0, -1.0 / self.schedule.sigma(t)


class VPred(Parameterization):
    def to_v(self, m, x, t):
        return m

    def coeffs(self, t):
        return self.schedule.alpha_prime(t), self.schedule.sigma_prime(t)
