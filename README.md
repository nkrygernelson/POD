# Main Structure
This codebase follows an odrinary ML-research type repo with a source directory than can be run as a mdoule. The project allows for modularity with Hydra. The best way to quickly get an understanding of the project is to look at the base.yaml config file.

That said we present the most structurally important files and classes here:
1. `Parameterization` and the subclasses (in diffusion/parameterizations/paramv2)
2. `DiffusionProcess` (diffusion/diffusionprocess.py)
3. `TimeParam` (diffusion/time/time.py)
4. schedule scripts defined in the diffusion/schedules directory
and of course there are other important classes and scripts such train.py and generation.py which are relatively self-explanatory.
## `Parameterization` 
This is the base class that is inherited by the subclasses (`ScorePred`, `CleanPred`, `Vpred`, `NoisePred`). The main class defines methods that are shared by all subclasses representing a given parameterization, namely:
1. `training_target`
2. `to_x_0`
3. `to_noise`
4. `to_score`
5. `to_v`
6. `rev_sde`
7. `pfode`

This is an important interface between the network output and the rest of the codebase. When we specify which parameterization to train our model in (configured in the hydra configs), we instantiate the corresponding parameterization class which then defines all the methods that allow us to to train our model (training_target), convert between parameterizations and define the  PF-ode and rev-sde which can be used by a solver.
The following text is what I wrote in the docstring of the paramv2.py file and explains how the conversion works.

$$
\begin{bmatrix} x_t \\ \text{model\_out} \end{bmatrix}
=
\begin{bmatrix} \alpha_t & \sigma_t \\ a & b \end{bmatrix}
\begin{bmatrix} x_0 \\ \epsilon \end{bmatrix}
$$

where the coefficients $a$ and $b$ are determined by the parameterization:

| Parameterization | $a$        | $b$           |
|------------------|------------|---------------|
| noise            | $0$        | $1$           |
| $x_0$            | $1$        | $0$           |
| score            | $0$        | $-1/\sigma_t$ |
| v                | $\alpha_t'$| $\sigma_t'$   |

The inverse transformation is found by inverting the $2 \times 2$ matrix:

$$
\begin{bmatrix} x_0 \\ \epsilon \end{bmatrix}
=
\frac{1}{\det}
\begin{bmatrix} b & -\sigma_t \\ -a & \alpha_t \end{bmatrix}
\begin{bmatrix} x_t \\ \text{model\_out} \end{bmatrix}
\qquad \det = \alpha_t b - \sigma_t a
$$

This is only defined when the determinant is non-zero. The stability of the
conversion can be monitored by tracking the magnitude of the determinant.
## DiffusionProcess

This class orchestrates a number of different components to produce the training loss and use the network and parameterization instance to predicti across parameterizations. It uses the time distributions, weighting functions, x_0 samples, network and parameterization-specific training target to produce a training loss.
The methods of DiffusionProcess are:
- `forward_sample`
- `training_loss`
- `predict_noise`
- `predict_x_0`
- `predict_score`
- `predict_v`
## `TimeParam`

This class allows for us to replce the canoncial time t from the schedule with a function $u =f(t)$ that we can define. This class lets us define a bijection from $t$ to $u$. We feed the schedule and parameterization methods $t$ while feeding the model the new time $u$. THe time grid used for solvers can be sampled from a distribution in u and coverted to t.

# TrainProc
THe codebase allows for a few different training procedures that allow us to perform reflow, standard traing oor classifier-free guidance.

# solvers
The implemented solvers are:
- DDIM
- DEIS 
- Euler (Mauryama)

# Documentation
The code is reasonably well commented/readable. I have used claude to generate very complete documentation in docs/Readme.md. It is porbably not very readable but it is certainly quite complete.
