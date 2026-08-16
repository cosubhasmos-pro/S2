#!/usr/bin/env python
"""
MOND orbit fit for S2 around Sgr A*, using emcee.
================================================================================
PIPELINE
================================================================================
  1. Load data, print diagnostic ranges. Catches unit/scale problems in
     seconds, before any expensive fitting.
  2. Global search (differential_evolution) over wide, physically-complete
     bounds, using loose integration tolerance for speed. This replaces
     hand-tuning a narrow prior box around a guessed starting point - it
     finds the actual best fit directly.
  3. Diagnostic overlay + normalized residuals, using the global best fit.
     Check this BEFORE running MCMC: if the model doesn't visibly trace the
     data (especially the RV panel, which is the most sensitive check),
     nothing downstream will fix it - go back and check the model/data
     conventions instead of tuning priors.
  4. emcee MCMC, walkers seeded in a small ball around the global best fit,
     run until autocorrelation-time convergence is actually detected
     (rather than a fixed, guessed step count).
  5. Corner plot, correctly labeled in the units the parameters are
     actually fit in.
================================================================================
"""

import time
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


# ==============================================================================
# STEP 1: load data, print diagnostics
# ==============================================================================
def load_data():
    """
    Loads astrometry + RV data via the project's S2DataLoader and packs
    everything the rest of the pipeline needs into one dict. Printing the
    ranges immediately is deliberate: unit/scale bugs (wrong arcsec/AU
    conversion, wrong km/s conversion, etc.) show up here in one glance,
    long before an expensive fit would otherwise reveal them.
    """
    loader = S2DataLoader()
    loader.load_data()
    data = loader.get_data()
    t_ast, RA_obs, Dec_obs = data['t_ast'], data['ra_obs'], data['dec_obs']
    sigma_RA, sigma_Dec = data['ra_err'], data['dec_err']
    t_rv, RV_obs, sigma_RV = data['t_rv'], data['rv_obs'], data['rv_err']
    n_ast = len(t_ast)
    t_obs = np.concatenate([t_ast, t_rv])

    print("=" * 70)
    print("DATA DIAGNOSTICS")
    print("=" * 70)
    print(f"n_astrometric = {n_ast}, n_RV = {len(t_rv)}")
    print(f"t_ast  range: [{t_ast.min():.3f}, {t_ast.max():.3f}]")
    print(f"t_rv   range: [{t_rv.min():.3f}, {t_rv.max():.3f}]")
    print(f"RA_obs range: [{RA_obs.min():.4f}, {RA_obs.max():.4f}] arcsec")
    print(f"Dec_obs range: [{Dec_obs.min():.4f}, {Dec_obs.max():.4f}] arcsec")
    print(f"RV_obs range: [{RV_obs.min():.1f}, {RV_obs.max():.1f}] km/s")
    print("=" * 70)

    return dict(t_ast=t_ast, RA_obs=RA_obs, Dec_obs=Dec_obs,
                sigma_RA=sigma_RA, sigma_Dec=sigma_Dec,
                t_rv=t_rv, RV_obs=RV_obs, sigma_RV=sigma_RV,
                n_ast=n_ast, t_obs=t_obs)


# ==============================================================================
# STEP 2 setup: likelihood, prior, bounds
# ==============================================================================
def make_log_likelihood(D):
    """
    Returns a log_likelihood(theta, rtol, atol) closure over the data dict.
    rtol/atol are exposed so the search phase can use loose (fast)
    tolerances while the final MCMC uses tight (accurate) ones - see
    mond_model.py's docstring for why that distinction matters here.
    """
    def log_likelihood(theta, rtol=1e-10, atol=1e-12):
        try:
            RA_mod, Dec_mod, RV_mod = orbit_model_mond(
                D['t_obs'], theta, D['n_ast'], rtol=rtol, atol=atol)
            if len(RA_mod) != len(D['RA_obs']) or len(RV_mod) != len(D['RV_obs']):
                return -np.inf
            chi2 = (np.sum(((RA_mod - D['RA_obs']) / D['sigma_RA']) ** 2) +
                    np.sum(((Dec_mod - D['Dec_obs']) / D['sigma_Dec']) ** 2) +
                    np.sum(((RV_mod - D['RV_obs']) / D['sigma_RV']) ** 2))
            return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
        except Exception:
            return -np.inf
    return log_likelihood


def make_bounds(T0_lo, T0_hi):
    """
    Physically-complete bounds. Angles cover their FULL range (0-180 for i,
    0-360 for omega/Omega) so no convention or degeneracy issue can hide
    outside the search box. a, M, a0 are wide enough to comfortably contain
    the real S2 system. T0 spans the actual data baseline - not a
    hand-guessed narrow window - since pericenter must fall somewhere the
    data can constrain it.
    """
    return [
        (0.3, 3.0),      # a, 1e3 AU
        (0.01, 0.98),    # e
        (0.0, 180.0),    # i, deg
        (0.0, 360.0),    # omega, deg
        (0.0, 360.0),    # Omega, deg
        (T0_lo, T0_hi),  # T0, yr
        (1.0, 8.0),      # M, 1e6 Msun
        (0.001, 0.5),    # a0
    ]


def log_prior(theta, bounds):
    for val, (lo, hi) in zip(theta, bounds):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def make_log_posterior(log_likelihood, bounds):
    def log_posterior(theta):
        lp = log_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)  # tight tolerances by default - final answer
        return lp + ll if np.isfinite(ll) else -np.inf
    return log_posterior


# ==============================================================================
# STEP 2: global search
# ==============================================================================
def run_global_search(log_likelihood, bounds, maxiter=150, popsize=15,
                       seed=42, label="search"):
    """
    Fast, serial, loose-tolerance differential_evolution search with live
    progress printed every 10 generations. Serial (workers=1) is
    deliberate: multiprocessing here previously caused silent, hard-to-
    debug hangs with no benefit given how cheap each loose-tolerance
    likelihood call already is.
    """
    def neg_ll_fast(theta):
        ll = log_likelihood(theta, rtol=1e-8, atol=1e-9)
        return -ll if np.isfinite(ll) else 1e10

    gen = [0]
    t0 = time.time()

    def progress(xk, convergence):
        gen[0] += 1
        if gen[0] % 10 == 0:
            print(f"  [{label}] gen {gen[0]:4d}  convergence={convergence:.2e}  "
                  f"elapsed={time.time()-t0:6.1f}s", flush=True)

    result = differential_evolution(
        neg_ll_fast, bounds, maxiter=maxiter, popsize=popsize, tol=1e-6, seed=seed,
        workers=1, polish=True, disp=False, callback=progress,
    )
    return result.x, result.fun


def check_mirror_degeneracy(log_likelihood, bounds, theta_ref, window=20.0,
                             maxiter=80, popsize=12):
    """
    Targeted check for the sky-plane mirror degeneracy: astrometric
    position depends on inclination only through cos(i), while radial
    velocity depends on sin(i) and the sense of orbital motion. This means
    i and (180-i) can trace nearly the same sky-plane ellipse while
    predicting oppositely-behaved RV curves. A search dominated by many
    more astrometric residuals than RV residuals can settle into the
    wrong one with little pressure to leave.

    Rather than hoping a random restart stumbles onto the correct branch,
    explicitly build the mirror candidate (i -> 180-i, other angles free
    to readjust) and locally refine it - this converges much faster than a
    cold global search because it starts near a good region already.
    Returns whichever of {theta_ref, mirror-refined} has the better TRUE
    (tight-tolerance) likelihood.
    """
    i_mirror = 180.0 - theta_ref[2]
    print(f"\nChecking mirror-degeneracy candidate: i={theta_ref[2]:.2f} -> "
          f"i_mirror={i_mirror:.2f} (other angles re-optimized around it)")

    mirror_bounds = list(bounds)
    mirror_bounds[2] = (max(0.0, i_mirror - window), min(180.0, i_mirror + window))
    # give omega/Omega room to readjust to compensate, but seeded near the
    # reference solution's values so this stays a LOCAL, fast refinement
    for j in (3, 4):
        lo, hi = bounds[j]
        mirror_bounds[j] = (max(lo, theta_ref[j] - 60), min(hi, theta_ref[j] + 60))

    def neg_ll_fast(theta):
        ll = log_likelihood(theta, rtol=1e-8, atol=1e-9)
        return -ll if np.isfinite(ll) else 1e10

    result = differential_evolution(
        neg_ll_fast, mirror_bounds, maxiter=maxiter, popsize=popsize, tol=1e-6,
        seed=99, workers=1, polish=True, disp=False,
    )
    ll_mirror = log_likelihood(result.x)
    ll_ref = log_likelihood(theta_ref)
    print(f"  reference: i={theta_ref[2]:.2f}  true_logL={ll_ref:.2f}")
    print(f"  mirror:    i={result.x[2]:.2f}  true_logL={ll_mirror:.2f}")

    if ll_mirror > ll_ref:
        print("  -> mirror candidate is better, using it")
        return result.x, -ll_mirror
    print("  -> reference solution stands")
    return theta_ref, -ll_ref


# ==============================================================================
# STEP 3: diagnostic overlay + residuals
# ==============================================================================
def diagnostic_plot(D, theta_best, outpath='bestfit_diagnostic.pdf'):
    """
    Overlays the best-fit model on the data (sky-plane track, RA/Dec/RV vs
    time) and reports normalized residuals. The RV panel is the most
    sensitive diagnostic in this problem: RV depends on orbital phase far
    more sharply than position does, so a wrong T0 (or any other real
    structural problem) shows up there first and most clearly.
    """
    RA_best, Dec_best, RV_best = orbit_model_mond(D['t_obs'], theta_best, D['n_ast'])
    t_ast, RA_obs, Dec_obs = D['t_ast'], D['RA_obs'], D['Dec_obs']
    sigma_RA, sigma_Dec = D['sigma_RA'], D['sigma_Dec']
    t_rv, RV_obs, sigma_RV = D['t_rv'], D['RV_obs'], D['sigma_RV']

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
    plt.savefig(outpath)
    plt.close(fig)

    rms_ra = np.std((RA_best - RA_obs) / sigma_RA)
    rms_dec = np.std((Dec_best - Dec_obs) / sigma_Dec)
    rms_rv = np.std((RV_best - RV_obs) / sigma_RV)
    print(f"\nRMS normalized residuals: RA={rms_ra:.2f}, Dec={rms_dec:.2f}, RV={rms_rv:.2f}")
    print("(want ~1 if the model and error bars are both right)")
    if max(rms_ra, rms_dec, rms_rv) > 10:
        print("*** WARNING: residuals are far from 1 - inspect the plot before ***")
        print("*** running MCMC. Do not just proceed and hope. ***")
    return rms_ra, rms_dec, rms_rv


# ==============================================================================
# STEP 4: emcee
# ==============================================================================
def run_mcmc(log_posterior, bounds, theta_best, nwalkers=32, max_n=20000):
    """
    Seeds walkers in a small ball around the global best fit (not a hand
    guess), then runs until the standard emcee auto-convergence criterion
    is met: iteration count > 50x the autocorrelation time, and tau itself
    has stabilized to within 1% between checks. This replaces guessing a
    fixed step count, which was unreliable in both directions (too short
    to converge, or wastefully long).
    """
    ndim = len(theta_best)
    spread = np.array([hi - lo for lo, hi in bounds]) * 1e-3
    pos = theta_best + spread * np.random.randn(nwalkers, ndim)
    for j, (lo, hi) in enumerate(bounds):
        pos[:, j] = np.clip(pos[:, j], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    print("\nRunning emcee until autocorrelation-time convergence (or max_n cap)...")
    index, old_tau = 0, np.inf
    autocorr = np.empty(max_n)
    t0 = time.time()

    for _ in sampler.sample(pos, iterations=max_n, progress=False):
        if sampler.iteration % 100:
            continue
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        print(f"  step {sampler.iteration:6d}  mean_tau={np.mean(tau):8.1f}  "
              f"elapsed={time.time()-t0:7.1f}s", flush=True)
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
    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    print(f"Autocorrelation times: {dict(zip(PARAM_NAMES, np.round(tau_final, 1)))}")
    print(f"burn-in={burn}, thin={thin}, mean acceptance={np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Final sample count: {samples.shape[0]}")
    return samples


# ==============================================================================
# STEP 5: corner plot
# ==============================================================================
def make_corner_plot(samples, outpath='corner-mond_final.pdf'):
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
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    D = load_data()
    T0_LO, T0_HI = D['t_obs'].min(), D['t_obs'].max()
    bounds = make_bounds(T0_LO, T0_HI)
    log_likelihood = make_log_likelihood(D)

    print("\n--- Step 2: global search ---")
    theta_rough, _ = run_global_search(log_likelihood, bounds)
    print("\nRough best-fit theta:")
    for name, val, (lo, hi) in zip(PARAM_NAMES, theta_rough, bounds):
        frac = (val - lo) / (hi - lo)
        flag = "  <-- AT BOUNDARY, widen this bound and rerun" if (frac < 0.02 or frac > 0.98) else ""
        print(f"  {name:7s} = {val:10.4f}   (bounds {lo}-{hi}){flag}")

    print("\n--- Step 2b: mirror-degeneracy check (i -> 180-i) ---")
    theta_best, negll_best = check_mirror_degeneracy(log_likelihood, bounds, theta_rough)
    print("\nGlobal best-fit theta:")
    for name, val, (lo, hi) in zip(PARAM_NAMES, theta_best, bounds):
        frac = (val - lo) / (hi - lo)
        flag = "  <-- AT BOUNDARY, widen this bound and rerun" if (frac < 0.02 or frac > 0.98) else ""
        print(f"  {name:7s} = {val:10.4f}   (bounds {lo}-{hi}){flag}")
    print(f"  -log_likelihood = {negll_best:.2f}")

    print("\n--- Step 3: diagnostic overlay (check this BEFORE running MCMC) ---")
    diagnostic_plot(D, theta_best)

    print("\n--- Step 4: emcee ---")
    log_posterior = make_log_posterior(log_likelihood, bounds)
    samples = run_mcmc(log_posterior, bounds, theta_best)

    print("\n--- Step 5: corner plot ---")
    make_corner_plot(samples)

