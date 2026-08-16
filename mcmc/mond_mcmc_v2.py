#!/usr/bin/env python
"""
MOND orbit fit for S2 around Sgr A*, using emcee.

Pipeline (each step is automatic - no manual prior-tuning loop):
  1. Load data, print diagnostic ranges (catches unit/scale problems here,
     in seconds, instead of after a multi-hour MCMC run).
  2. Global optimization (differential_evolution) over WIDE, physically-
     complete bounds to find the true best fit, instead of hand-guessing
     a narrow prior box around a literature/hand guess.
  3. Overlay the best fit on the data and print normalized residuals.
     If this doesn't look right, nothing downstream will either - fix it
     here before spending compute on MCMC.
  4. Run emcee, initializing walkers around the GLOBAL best fit (not a
     hand guess), and let it run until autocorrelation-time convergence
     is actually detected, rather than a fixed guessed step count.
  5. Corner plot with correctly-labeled units.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from scipy.optimize import differential_evolution

from mond_model import orbit_model_mond
from data_loader import S2DataLoader

PARAM_NAMES = ["a", "e", "i", "omega", "Omega", "T0", "M", "a0"]
PARAM_LABELS = [r"$a$ [$10^3$ AU]", r"$e$", r"$i$ [deg]", r"$\omega$ [deg]",
                r"$\Omega$ [deg]", r"$T_0$ [yr]", r"$M$ [$10^6\,M_\odot$]", r"$a_0$"]

# ---------------------------------------------------------------------------
# STEP 1: load data and print diagnostics immediately
# ---------------------------------------------------------------------------
loader = S2DataLoader()
loader.load_data()
data = loader.get_data()

t_ast, RA_obs, Dec_obs = data['t_ast'], data['ra_obs'], data['dec_obs']
sigma_RA, sigma_Dec = data['ra_err'], data['dec_err']
t_rv, RV_obs, sigma_RV = data['t_rv'], data['rv_obs'], data['rv_err']
n_ast = len(t_ast)
t_obs = np.concatenate([t_ast, t_rv])

print("=" * 70)
print("DATA DIAGNOSTICS (check these look right before trusting anything else)")
print("=" * 70)
print(f"n_astrometric = {n_ast}, n_RV = {len(t_rv)}")
print(f"t_ast  range: [{t_ast.min():.3f}, {t_ast.max():.3f}]")
print(f"t_rv   range: [{t_rv.min():.3f}, {t_rv.max():.3f}]")
print(f"RA_obs range: [{RA_obs.min():.4f}, {RA_obs.max():.4f}]  (expect ~0.1-0.3 arcsec scale for S2)")
print(f"Dec_obs range: [{Dec_obs.min():.4f}, {Dec_obs.max():.4f}]")
print(f"RV_obs range: [{RV_obs.min():.1f}, {RV_obs.max():.1f}]  (expect ~hundreds to ~few 1000s km/s)")
print("=" * 70)

# T0 (pericenter time) must be searched over the ACTUAL data time span, not
# a hand-guessed tiny window - this was silently wrong before (prior was
# +/-0.05 assuming t was already ~0-centered, with no check that it was).
t_all = np.concatenate([t_ast, t_rv])
T0_LO, T0_HI = t_all.min(), t_all.max()
print(f"T0 search range set from data: [{T0_LO:.2f}, {T0_HI:.2f}]")

# ---------------------------------------------------------------------------
# STEP 2: likelihood, prior (WIDE + physically complete), and global fit
# ---------------------------------------------------------------------------
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

rms_ra = np.std((RA_best - RA_obs) / sigma_RA)
rms_dec = np.std((Dec_best - Dec_obs) / sigma_Dec)
rms_rv = np.std((RV_best - RV_obs) / sigma_RV)
print(f"\nRMS normalized residuals: RA={rms_ra:.2f}, Dec={rms_dec:.2f}, RV={rms_rv:.2f}")
print("(want ~1 if the model and error bars are both right; >>1 means a real")
print(" structural problem remains - check bestfit_diagnostic.pdf before proceeding)")

if max(rms_ra, rms_dec, rms_rv) > 10:
    print("\n*** WARNING: residuals are far from 1. Something is still wrong ***")
    print("*** structurally. Inspect bestfit_diagnostic.pdf before running MCMC. ***")

# ---------------------------------------------------------------------------
# STEP 4: emcee, initialized around the GLOBAL best fit, run to convergence
# ---------------------------------------------------------------------------
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
max_n = 20000
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
