#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import newton
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.integrate import solve_ivp
# Load the data which you just simulated
from data_loader import S2DataLoader
import tqdm


# # Load S2 Data

# In[2]:


# Load real S2 data
s2_data = S2DataLoader()
astro_df, vel_df = s2_data.load_data()
print(astro_df.head(3))
print(vel_df.head(3))

# Plot S2 data
fig = s2_data.plot_all()
fig.savefig('s2_data.pdf')
plt.show()


# In[3]:


data = s2_data.get_data()
t_ast = data['t_ast']
RA_obs = data['ra_obs']
Dec_obs = data['dec_obs']
sigma_RA = data['ra_err']
sigma_Dec = data['dec_err']
t_rv = data['t_rv']
RV_obs = data['rv_obs']  # km/s
sigma_RV = data['rv_err']  # km/s


# In[4]:


t_obs = np.concatenate([data['t_ast'], data['t_rv']])


# In[5]:


t_all = np.concatenate([t_ast, t_rv])
T0_LO, T0_HI = t_all.min(), t_all.max()
print(f"T0 search range set from data: [{T0_LO:.2f}, {T0_HI:.2f}]")


# In[ ]:





# # The Model

# In[6]:


KEPLER_G = 4 * np.pi**2          # GM_sun in AU^3/yr^2 (Kepler's constant)
MASS_UNIT = 1e6                  # theta's M is in units of 1e6 Msun
LENGTH_UNIT_AU = 1000.0          # theta's a is in units of 1000 AU
R0_PC = 8178.0                   # distance to Sgr A*, GRAVITY Collab. 2019 (pc)
AU_PER_YR_TO_KMS = 4.740470446   # exact conversion: 1 AU/yr in km/s


def mond_orbit_equations(t, y, GM, a0):
    """MOND EOM in the orbital plane (2D), fully physical units:
    positions in AU, velocities in AU/yr, GM in AU^3/yr^2, a0 in AU/yr^2."""
    x, y_pos, vx, vy = y
    r = np.sqrt(x**2 + y_pos**2)
    if r < 1e-8:
        return [0.0, 0.0, 0.0, 0.0]
    a_N = GM / r**2
    a_mond = (a_N + np.sqrt(a_N**2 + 4 * a_N * a0)) / 2  # simple mu(x)=x/(1+x)
    ax = -a_mond * (x / r)
    ay = -a_mond * (y_pos / r)
    return [vx, vy, ax, ay]


def orbit_model_mond(t, theta, n_ast):
    """
    MOND orbit model with full 3D position + velocity projection.

    Parameters
    ----------
    t : array-like
        Observation times, years. MUST be [t_ast..., t_rv...] concatenated,
        i.e. the first n_ast entries are astrometric epochs and the rest
        are RV epochs (matches how the data loader / t_obs is built).
    theta : array
        [a, e, i_deg, omega_deg, Omega_deg, T0, M, a0] - see module docstring.
    n_ast : int
        Number of astrometric epochs at the start of `t` (explicit, not
        inferred from a global RA_obs - this was a source of silent bugs
        when the model was called with a differently-sized time array).

    Returns
    -------
    RA_model, Dec_model : np.ndarray, length n_ast, arcsec
    RV_model : np.ndarray, length len(t)-n_ast, km/s
    """
    a, e, i_deg, omega_deg, Omega_deg, T0, M, a0 = theta
    n = len(t)

    if not (0 < e < 1 and a > 0 and M > 0 and a0 > 0 and 0 <= i_deg <= 180):
        return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)

    i = np.radians(i_deg)
    omega = np.radians(omega_deg)
    Omega = np.radians(Omega_deg)

    a_AU = a * LENGTH_UNIT_AU
    GM = KEPLER_G * (M * MASS_UNIT)

    r_peri = a_AU * (1 - e)
    v_peri = np.sqrt(GM * (1 + e) / (a_AU * (1 - e)))  # Newtonian vis-viva at pericenter
    y0 = [r_peri, 0.0, 0.0, v_peri]

    t_rel = np.asarray(t) - T0
    unique_t_rel, inv_idx = np.unique(t_rel, return_inverse=True)
    t_span = (unique_t_rel.min() - 0.1, unique_t_rel.max() + 0.1)

    try:
        sol = solve_ivp(
            mond_orbit_equations, t_span, y0, args=(GM, a0),
            t_eval=np.sort(unique_t_rel), method='DOP853',
            rtol=1e-10, atol=1e-10,
        )
    except Exception:
        return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)

    if not sol.success:
        return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)

    x_orb = sol.y[0][inv_idx]
    y_orb = sol.y[1][inv_idx]
    vx_orb = sol.y[2][inv_idx]
    vy_orb = sol.y[3][inv_idx]

    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_o, sin_o = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)

    X = (cos_O*cos_o - sin_O*sin_o*cos_i)*x_orb + (-cos_O*sin_o - sin_O*cos_o*cos_i)*y_orb
    Y = (sin_O*cos_o + cos_O*sin_o*cos_i)*x_orb + (-sin_O*sin_o + cos_O*cos_o*cos_i)*y_orb
    vX = (cos_O*cos_o - sin_O*sin_o*cos_i)*vx_orb + (-cos_O*sin_o - sin_O*cos_o*cos_i)*vy_orb
    vY = (sin_O*cos_o + cos_O*sin_o*cos_i)*vx_orb + (-sin_O*sin_o + cos_O*cos_o*cos_i)*vy_orb
    vZ = (sin_o*sin_i)*vx_orb + (cos_o*sin_i)*vy_orb

    RA_model = X / R0_PC                   # AU -> arcsec
    Dec_model = Y / R0_PC                  # AU -> arcsec
    RV_model = vZ * AU_PER_YR_TO_KMS        # AU/yr -> km/s

    return RA_model[:n_ast], Dec_model[:n_ast], RV_model[n_ast:]


# In[7]:


def log_likelihood(theta):
    try:
        RA_mod, Dec_mod, RV_mod = orbit_model_mond(t_obs, theta, n_ast)
        if len(RA_mod) != len(RA_obs) or len(RV_mod) != len(RV_obs):
            return -np.inf
        chi2 = (np.sum(((RA_mod - RA_obs) / sigma_RA) ** 2) +
                np.sum(((Dec_mod - Dec_obs) / sigma_Dec) ** 2) +
                np.sum(((RV_mod - RV_obs) / sigma_RV) ** 2))
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2
    except Exception:
        return -np.inf


# In[8]:


# Physically-complete bounds. Angles cover their FULL range so no
# convention/degeneracy issue can hide outside the search box. a, M, a0
# are wide enough to comfortably contain the real S2 system.
BOUNDS = [
    (0.3, 3.0),          # a, 1e3 AU
    (0.01, 0.98),        # e
    (0.0, 180.0),        # i, deg
    (0.0, 360.0),        # omega, deg
    (0.0, 360.0),        # Omega, deg
    (T0_LO, T0_HI),      # T0, yr - data-driven, not hand-guessed
    (1.0, 8.0),          # M, 1e6 Msun
    (0.001, 0.5),        # a0 - see mond_model.py docstring re: physical units
]

def log_prior(theta):
    for val, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# In[ ]:





# In[ ]:


from scipy.optimize import differential_evolution
PARAM_NAMES = ["a", "e", "i", "omega", "Omega", "T0", "M", "a0"]
PARAM_LABELS = [r"$a$ [$10^3$ AU]", r"$e$", r"$i$ [deg]", r"$\omega$ [deg]",
                r"$\Omega$ [deg]", r"$T_0$ [yr]", r"$M$ [$10^6\,M_\odot$]", r"$a_0$"]

t_ast, RA_obs, Dec_obs = data['t_ast'], data['ra_obs'], data['dec_obs']
sigma_RA, sigma_Dec = data['ra_err'], data['dec_err']
t_rv, RV_obs, sigma_RV = data['t_rv'], data['rv_obs'], data['rv_err']
n_ast = len(t_ast)
t_obs = np.concatenate([t_ast, t_rv])

def neg_log_likelihood(theta):
    ll = log_likelihood(theta)
    return -ll if np.isfinite(ll) else 1e10


print("\nRunning global optimizer (differential_evolution) over wide bounds...")
result = differential_evolution(
    neg_log_likelihood, BOUNDS,
    maxiter=300, popsize=20, tol=1e-8, seed=42,
    workers=-1, polish=True, disp=False,
)
theta_best = result.x

print("\nGlobal best-fit theta:")
for name, val, (lo, hi) in zip(PARAM_NAMES, theta_best, BOUNDS):
    frac = (val - lo) / (hi - lo)
    flag = "  <-- AT BOUNDARY, widen this bound and rerun" if (frac < 0.02 or frac > 0.98) else ""
    print(f"  {name:7s} = {val:10.4f}   (bounds {lo}-{hi}){flag}")
print(f"  -log_likelihood = {result.fun:.2f}")


# In[16]:


# ---------------------------------------------------------------------------
# STEP 3: overlay + residual diagnostic - DO NOT SKIP THIS.
# If residuals show structure, no amount of MCMC tuning fixes it; go back
# and check the model/data conventions instead.
# ---------------------------------------------------------------------------
RA_best, Dec_best, RV_best = orbit_model_mond(t_obs, theta_best, n_ast)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].errorbar(RA_obs, Dec_obs, xerr=sigma_RA, yerr=sigma_Dec, fmt='o',
                     color='black', markersize=3, alpha=0.5, label='data')
sort_idx = np.argsort(t_ast)
axes[0, 0].plot(RA_best[sort_idx], Dec_best[sort_idx], '-', color='crimson',
                 linewidth=1.2, alpha=0.8, label='best fit')
axes[0, 0].set(xlabel='RA (arcsec)', ylabel='Dec (arcsec)', title='Sky-plane track')
axes[0, 0].invert_xaxis()
axes[0, 0].legend()

axes[0, 1].errorbar(t_ast, RA_obs, yerr=sigma_RA, fmt='o', color='blue',
                     markersize=3, alpha=0.5, label='data')
axes[0, 1].plot(t_ast, RA_best, '.', color='crimson', markersize=4, label='model')
axes[0, 1].set(xlabel='Time (yr)', ylabel='RA (arcsec)', title='RA vs time')
axes[0, 1].legend()

axes[1, 0].errorbar(t_ast, Dec_obs, yerr=sigma_Dec, fmt='o', color='green',
                     markersize=3, alpha=0.5, label='data')
axes[1, 0].plot(t_ast, Dec_best, '.', color='crimson', markersize=4, label='model')
axes[1, 0].set(xlabel='Time (yr)', ylabel='Dec (arcsec)', title='Dec vs time')
axes[1, 0].legend()

axes[1, 1].errorbar(t_rv, RV_obs, yerr=sigma_RV, fmt='o', color='purple',
                     markersize=3, alpha=0.5, label='data')
axes[1, 1].plot(t_rv, RV_best, '.', color='crimson', markersize=4, label='model')
axes[1, 1].set(xlabel='Time (yr)', ylabel='RV (km/s)', title='RV vs time')
axes[1, 1].legend()
plt.tight_layout()
plt.savefig('bestfit_diagnostic.pdf')
plt.close(fig)


# 
# EMCEE
# 

# In[17]:


ndim = len(theta_best)
nwalkers = 32

# small ball around the global best fit, scaled to each parameter's range
spread = np.array([hi - lo for lo, hi in BOUNDS]) * 1e-3
pos = theta_best + spread * np.random.randn(nwalkers, ndim)
# clip to bounds so no walker starts outside the prior
for j, (lo, hi) in enumerate(BOUNDS):
    pos[:, j] = np.clip(pos[:, j], lo + 1e-6, hi - 1e-6)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

print("\nRunning emcee until autocorrelation-time convergence (or max_n cap)...")
max_n = 2000
index = 0
autocorr = np.empty(max_n)
old_tau = np.inf

for sample in sampler.sample(pos, iterations=max_n, progress=True):
    if sampler.iteration % 100:
        continue
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    converged = np.all(tau * 50 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        print(f"Converged at iteration {sampler.iteration}")
        break
    old_tau = tau
else:
    print(f"Reached max_n={max_n} without formal convergence - "
          f"treat results as provisional, consider raising max_n.")

tau_final = sampler.get_autocorr_time(tol=0)
burn = int(2 * np.max(tau_final))
thin = max(1, int(0.5 * np.min(tau_final)))
print(f"Autocorrelation times: {dict(zip(PARAM_NAMES, np.round(tau_final, 1)))}")
print(f"Using burn-in={burn}, thin={thin}")

samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
af = sampler.acceptance_fraction
print(f"Mean acceptance fraction: {np.mean(af):.3f} (want roughly 0.2-0.5)")
print(f"Final sample count: {samples.shape[0]}")


# In[19]:


# ---------------------------------------------------------------------------
# STEP 5: corner plot
# ---------------------------------------------------------------------------
fig = corner.corner(
    samples, labels=PARAM_LABELS, quantiles=[0.16, 0.5, 0.84],
    show_titles=True, title_fmt='.3f',
    title_kwargs={"fontsize": 10, "pad": 5}, label_kwargs={"fontsize": 10},
    color="#3498db", smooth=1.0, bins=40, plot_datapoints=False,
    fill_contours=True, levels=[0.68, 0.95],
    hist_kwargs={"density": True, "edgecolor": "k", "linewidth": 0.9,
                 "histtype": "stepfilled", "alpha": 0.3},
    contour_kwargs={"linewidths": 1.8, "linestyles": "solid"},
)
fig.set_size_inches(12, 12)
plt.tight_layout(pad=0.5)
plt.savefig('corner-mond_final.pdf')
plt.close(fig)
print("\nSaved: bestfit_diagnostic.pdf, corner-mond_final.pdf")


# ## Convergance Plots

# In[ ]:




