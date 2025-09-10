### Hamiltonian Monte Carlo Sampling

The sampler uses a Hamiltonian function

$$
H(\theta, p) = U(\theta) + K(p),
$$

where 

$$
U(\theta) = -\log p(\theta \mid \text{data})
$$

is the **potential energy** (negative log posterior), and

$$
K(p) = \frac{1}{2} p^\top M^{-1} p
$$

is the **kinetic energy**, with $p$ an auxiliary momentum and $M$ the mass matrix.  

The sampler simulates Hamiltonian dynamics:

$$
\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = M^{-1}p, 
\quad
\frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = -\nabla_\theta U(\theta),
$$

which in theory conserves the total energy $H(\theta,p)$. In practice, numerical integration introduces an **energy error**

$$
\Delta H = H(\theta', p') - H(\theta, p),
$$

and warmup/adaptation procedures aim to control this error.

**Warmup**

The warmup phase estimates and adapts the integrator step size $\varepsilon$ for Hamiltonian dynamics. 
Smaller step sizes correspond to more accurate numerical integration, ensuring smaller energy errors $\Delta H$.  

The discretized dynamics for one integration step are:

$$
\theta_{t+1} \approx \theta_t + \varepsilon M^{-1} p, \quad
p_{t+1} \approx p_t - \varepsilon \nabla_\theta U(\theta_t),
$$

where $U(\theta) = -\log p(\theta | \text{data})$ is the potential energy, and $M$ is the mass matrix.  

The total **trajectory length** in parameter space is given by $L$ (number of integration steps), with total distance traveled approximately $L \cdot \varepsilon$.

**make_L_step_size_adaptation**

The function *make_L_step_size_adaptation* adapts the trajectory length $L$ and integrator step size $\varepsilon$

To guide the step size adaptation, we define schedules for the desired energy variance $\Delta H$:
- *get_desired_energy_var_linear*: Linearly interpolates between start and end values.
- *get_desired_energy_var_exp*: Exponentially interpolates, decaying quickly at first.

For very large initial variance (*desired_energy_var_start* $> 2.0$), the exponential schedule is used to stablize the dynamics. Otherwise, the linear schedule is used.

The function *predictor* performs a single interation step, computes the resulting $\Delta H$, and updates the step size according to the difference between the observed energy variance and the target variance from the chosen schedule.

The *predictor* function takes the following arguments:

- *previous_state*: Is of type `MCLMCState` (from BlackJAX). It contains the current **position** and **momentum** state (pytree of parameter arrays).  
- *params*: Is of type `MCLMCAdaptationState`. It contains the current **adaptation parameters**:  
  - *L*: number of integration steps  
  - *step_size*: integrator step size  
  - *sqrt_diag_cov*: diagonal preconditioning vector  
  `params` describes the current **state of the dynamics**, not the position and momentum themselves.  
- *adaptive_state*: Is of type tuple `(time, x_average, step_size_max)`, where:  
  - `time` is a scalar float, keeping track of weighted time in the running average  
  - `x_average` is a scalar float, used to compute the adapted step size  
  - `step_size_max` is a scalar float, the maximum allowed step size to avoid divergences  
  `adaptive_state` is essentially used to **update the step size during warmup**.  
- *rng_key*: JAX PRNG key, used for randomness in the MCLMC kernel.  
- *step_number*: Integer scalar, the current iteration step in the warmup loop.  

The outputs of the *predictor* function are:

- *state*: Is of type `MCLMCState`, the **new sampler state** after one integration step.  
- *params_new*: Is of type `MCLMCAdaptationState`, which gives the **updated adaptation parameters** (step size, L, and preconditioning).  
- *adaptive_state*: Is of type tuple `(time, x_average, step_size_max)`, which gives the **updated streaming averages** for step size adaptation.  
- *success*: Boolean, indicates whether the integration step was **successful** (i.e., no NaNs or divergences occurred).  

Further, we introduce

$$
\xi = \frac{(\Delta H)^2}{\text{dim} \times \text{desired variance}} + 10^{-8},
$$

where $\Delta H$ is the energy change after one integration step, $\text{dim}$ is the number of parameters, and $\text{desired variance}$ is the target variance for $\Delta H$.  

- If $\Delta H$ matches the target variance, we have $\xi \approx 1$.  
- If $\Delta H$ is too large, $\xi > 1$.  
- If $\Delta H$ is too small, $\xi < 1$.  

We then define a `weight` used in the **weighted running average** for computing the new step size.  

- Large $\Delta H$ → small weight → this step has less influence on updating the step size.  
- Small $\Delta H$ → large weight → this step contributes more to the step size adaptation.  

The parameter `trust_in_estimate` controls how aggressively we trust the observed $\Delta H$:  
- Larger `trust_in_estimate` → broader weighting → more aggressive step size updates.  
- Smaller `trust_in_estimate` → conservative updates.



