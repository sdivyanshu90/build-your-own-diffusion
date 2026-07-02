# The mathematics, mapped to the code

This is the full derivation chain behind `diffusionlab`, with pointers to
the exact functions that implement each equation and the tests that verify
it. Notation follows Ho et al. (2020), *Denoising Diffusion Probabilistic
Models*.

## 1. The forward (noising) process

Fix a chain length `T` and a variance schedule `beta_1..beta_T` (we index
`t = 0..T-1` in code). The forward process gradually corrupts data:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)
```

Define `alpha_t = 1 - beta_t` and `alpha_bar_t = prod_{s<=t} alpha_s`.
A Gaussian composed with a Gaussian is Gaussian, so the t-step corruption
collapses to a single closed form:

```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,   eps ~ N(0, I)
```

**Code:** `BaseScheduler.add_noise` (`schedulers/base.py`).
**Tests:** `test_add_noise_interpolates_signal_and_noise`,
`test_add_noise_preserves_variance_of_unit_inputs` (checks
`alpha_bar + (1 - alpha_bar) = 1` numerically for all t).

As `t -> T`, `alpha_bar_t -> 0` and `x_t` becomes pure noise -- that is what
lets sampling *start* from `N(0, I)`.

### Beta schedules (`schedulers/schedules.py`)

- **linear**: `linspace(1e-4, 0.02, T)` -- the original DDPM schedule.
- **scaled_linear**: linear in `sqrt(beta)` -- Stable Diffusion's schedule.
- **cosine** (Nichol & Dhariwal, 2021): defined through the *signal* rather
  than the noise: `alpha_bar(t) = cos^2(((t/T + s)/(1 + s)) * pi/2)`,
  normalised so `alpha_bar(0) = 1`, and converted to betas via
  `beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}`, clipped at 0.999. It
  destroys information more gradually near both ends of the chain.

All schedules are computed in **float64**: the cumulative product of ~1000
numbers slightly below 1 loses visible precision in float32.

**Tests:** monotonicity, bounds, endpoint values, and an exact check that
the cosine betas reproduce the defining `alpha_bar(t)` formula.

## 2. The training objective

The reverse process is learned as
`p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), sigma_t^2 I)`. Maximising the
variational lower bound reduces (with the fixed-variance choice and Ho et
al.'s simplification) to a plain denoising MSE:

```
L_simple = E_{x_0, eps, t} [ || eps - eps_theta(x_t, t) ||^2 ]
```

with `t ~ Uniform{0..T-1}` and `x_t` from the closed form above. This is
exactly `Trainer._compute_loss` (`training/trainer.py`).

### Prediction parameterisations

The network can be trained to regress different targets; all are linked by
the forward equation, writing `a = sqrt(alpha_bar_t)`,
`b = sqrt(1 - alpha_bar_t)` (so `x_t = a x_0 + b eps`):

| `prediction_type` | Network target | Recover x0 from output `o` |
| --- | --- | --- |
| `epsilon` | `eps` | `x0 = (x_t - b o) / a` |
| `sample` | `x_0` | `x0 = o` |
| `v_prediction` | `v = a eps - b x_0` (Salimans & Ho, 2022) | `x0 = a x_t - b o` |

The v-prediction identity: substitute `eps = (x_t - a x_0)/b` into `v`:
`v = a (x_t - a x_0)/b - b x_0 = (a x_t - x_0 (a^2 + b^2))/b = (a x_t - x_0)/b`,
hence `x_0 = a x_t - b v` using `a^2 + b^2 = 1`.

**Code:** `training_target`, `predict_original_sample`, `predict_epsilon`,
`get_velocity` in `schedulers/base.py`.
**Tests:** `test_predict_original_sample_roundtrip` feeds the *true* target
back through the conversion for every parameterisation and every timestep
regime and requires exact recovery of `x_0`.

## 3. DDPM sampling (ancestral)

Bayes' rule on the forward process gives the *exact* posterior for adjacent
steps, which is Gaussian:

```
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_t(x_t, x_0), sigma_t^2 I)

mu_t   = [ beta_t sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t) ] x_0
       + [ (1 - alpha_bar_{t-1}) sqrt(alpha_t) / (1 - alpha_bar_t) ] x_t

sigma_t^2 = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)     ("fixed_small")
```

Sampling replaces the unknown `x_0` with the model's estimate
(`predict_original_sample`, optionally clipped to [-1, 1] -- valid because
real images live there, and clipping stabilises the early, mostly-noise
steps). One reverse step is then

```
x_{t-1} = mu_t(x_t, x0_hat) + sigma_t z,   z ~ N(0, I),   (no noise at t=0)
```

`variance_type="fixed_large"` uses `beta_t` instead of the posterior
variance -- the upper of the two variance choices discussed by Ho et al.

**Code:** `DDPMScheduler` (`schedulers/ddpm.py`); the posterior coefficients
are precomputed in float64 in `__init__`, `step` applies them.
**Tests:** coefficient identities, non-negative variance with finite
clipped log, determinism at `t=0`, generator reproducibility.

Because the posterior above is only defined between *adjacent* timesteps,
`DDPMScheduler.set_timesteps` rejects any subsampled chain.

## 4. DDIM sampling (accelerated)

Song et al. (2021) construct a *family* of non-Markovian processes that all
share the DDPM marginals `q(x_t | x_0)` -- and therefore the same training
objective. This is the key insight that lets one trained model be sampled
with far fewer steps. For consecutive elements `(t, t_prev)` of any
descending subsequence of `0..T-1`:

```
x_{t_prev} = sqrt(alpha_bar_prev) x0_hat
           + sqrt(1 - alpha_bar_prev - sigma_t^2) eps_hat        (direction)
           + sigma_t z                                            (noise)

sigma_t = eta sqrt( (1 - alpha_bar_prev) / (1 - alpha_bar_t) )
             sqrt( 1 - alpha_bar_t / alpha_bar_prev )
```

- `eta = 0`: fully deterministic; the sampler becomes a discretised ODE and
  a fixed `x_T` always maps to the same image.
- `eta = 1` over the full chain: `sigma_t^2` reduces *exactly* to the DDPM
  posterior variance. Proof: with adjacent steps,
  `alpha_bar_t = alpha_t alpha_bar_{t-1}`, so
  `1 - alpha_bar_t / alpha_bar_prev = 1 - alpha_t = beta_t`, giving
  `sigma_t^2 = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`. QED.
- At the final step `t_prev = -1`, `alpha_bar_prev := 1` and the update
  degenerates to `x_out = x0_hat` -- the chain ends on the model's clean
  estimate.

After clipping `x0_hat`, the code re-derives `eps_hat` from the *clipped*
estimate (`predict_epsilon`) so that `x_t`, `x0_hat`, and `eps_hat` remain
mutually consistent within the step.

**Code:** `DDIMScheduler` (`schedulers/ddim.py`).
**Tests:** `test_ddim_eta_one_variance_equals_ddpm_posterior` verifies the
identity above for every t; `test_ddim_final_step_returns_predicted_x0` and
`test_ddim_perfect_model_recovers_x0_in_one_step` verify the boundary
behaviour with an oracle model; determinism at `eta=0` is asserted both at
step level and end-to-end through the pipeline.

### Timestep spacing

`leading` picks `0, k, 2k, ...` (reversed), matching the original DDIM code;
`trailing` walks down from exactly `T-1`, which keeps the first sampling
step at the highest-noise level the model was trained on and empirically
helps at very low step counts (Lin et al., 2024, *Common Diffusion Noise
Schedules and Sample Steps are Flawed*).

## 5. EMA of weights

Sample quality is evaluated with an exponential moving average of the
parameters, `s <- d s + (1 - d) p` (`d = 0.9999` for long runs). The
average suppresses the last few thousand SGD steps' noise; empirically this
is worth a large chunk of FID on small datasets. **Code:** `training/ema.py`
(`lerp_` in-place update); the trainer swaps EMA weights in for previews and
the pipeline prefers them at load time.
