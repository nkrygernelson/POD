You have to make choices and while I tired to make this project as modular as possible, there are still
things that must be done the way that the code expects:

# Schedule, weights and t_distribution

I have a struggled a little with this design choice but have landed on this:
Schedules, weights and conversions expect time $t$ which is what is used in the schedule.
HCanging the time param in the hydra only changes what time param the model receives and the grid that the solvers receive.
Of course it possible to redefine the schedule in whatever time parameterization one wants without need to invoke time_param. This desgin is a little awakward because setting the time param to lognsr for isntance doesn't change the fact taht the schedule was defined in t so it is not a complete reparameterization.




