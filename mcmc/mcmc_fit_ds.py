import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from scipy.stats import norm
from data_loader import S2DataLoader
from mond_model_ds import orbit_model_mond   # corrected version

# --------------------------------------------
# 1. Load data
# --------------------------------------------
loader = S2DataLoader()
astro_df, vel_df = loader.load_data()
data = loader.get_data()

t_ast = data['t_ast']
ra_obs = data['ra_obs']
dec_obs = data['dec_obs']
ra_err = data['ra_err']
dec_err = data['dec_err']
t_rv = data['t_rv']
rv_obs = data['rv_obs']
rv_err = data['rv_err']

n_ast = len(t_ast)
t_all = np.concatenate([t_ast, t_rv])

# --------------------------------------------
# 2. Priors
# --------------------------------------------
def log_prior(theta):
    a, e, i_deg, omega_deg, Omega_deg, T0, M, a0 = theta
    # Uniform priors (physical ranges)
    if not (0.3 < a < 3.0):          # 1000 AU units
        return -np.inf
    if not (0.0 < e < 1.0):
        return -np.inf
    if not (0.0 < i_deg < 180.0):
        return -np.inf
    if not (0.0 < omega_deg < 360.0):
        return -np.inf
    if not (0.0 < Omega_deg < 360.0):
        return -np.inf
    if not (2000.0 < T0 < 2030.0):   # pericenter passage year
        return -np.inf
    if not (2.0 < M < 8.0):          # 1e6 Msun
        return -np.inf
    # log-uniform for a0 (in AU/yr^2)
    if not (1e-8 < a0 < 1e-4):
        return -np.inf
    return 0.0   # uniform prior => constant

# --------------------------------------------
# 3. Likelihood
# --------------------------------------------
def log_likelihood(theta):
    try:
        RA_model, Dec_model, RV_model = orbit_model_mond(t_all, theta, n_ast)
    except Exception:
        return -np.inf

    # Astrometry: RA and Dec
    chi2_ra = np.sum(((RA_model - ra_obs) / ra_err) ** 2)
    chi2_dec = np.sum(((Dec_model - dec_obs) / dec_err) ** 2)
    # RV
    chi2_rv = np.sum(((RV_model - rv_obs) / rv_err) ** 2)
    chi2 = chi2_ra + chi2_dec + chi2_rv
    return -0.5 * chi2

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# --------------------------------------------
# 4. Initial guess and MCMC setup
# --------------------------------------------
# A reasonable initial guess (S2-like)
theta0 = np.array([0.82, 0.88, 134.0, 66.0, 228.0, 2018.35, 4.0, 1e-6])
ndim = len(theta0)
nwalkers = 32
nsteps = 5000
burnin = 1000

# Initialize walkers in a small ball around theta0
pos = theta0 + 1e-4 * np.random.randn(nwalkers, ndim)
# Ensure positive a0
pos[:, -1] = np.abs(pos[:, -1])

# --------------------------------------------
# 5. Run MCMC
# --------------------------------------------
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, nsteps, progress=True)

# Flatten chain and discard burn-in
chain = sampler.get_chain(flat=True, discard=burnin)
samples = chain.reshape(-1, ndim)

# Best-fit (median)
theta_med = np.median(samples, axis=0)
print("Median parameters:")
labels = ['a (1000 AU)', 'e', 'i (deg)', 'omega (deg)', 'Omega (deg)', 'T0 (yr)', 'M (1e6 Msun)', 'a0 (AU/yr^2)']
for lab, val in zip(labels, theta_med):
    print(f"{lab}: {val:.4f}")

# --------------------------------------------
# 6. Corner plot
# --------------------------------------------
fig = corner.corner(samples, labels=labels, truths=theta0, truth_color='red',
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('corner_plot.png', dpi=300)
plt.close(fig)

# --------------------------------------------
# 7. Best-fit model plot
# --------------------------------------------
RA_best, Dec_best, RV_best = orbit_model_mond(t_all, theta_med, n_ast)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Sky position
ax = axs[0,0]
ax.errorbar(ra_obs, dec_obs, xerr=ra_err, yerr=dec_err, fmt='o', color='k', alpha=0.5, label='Data')
ax.plot(RA_best, Dec_best, 'r-', lw=2, label='Best-fit MOND')
ax.set_xlabel('RA (arcsec)')
ax.set_ylabel('Dec (arcsec)')
ax.invert_xaxis()
ax.legend()

# RA vs time
ax = axs[1,0]
ax.errorbar(t_ast, ra_obs, yerr=ra_err, fmt='o', color='k', alpha=0.5, label='Data')
ax.plot(t_ast, RA_best, 'r-', lw=2)
ax.set_xlabel('Time (yr)')
ax.set_ylabel('RA (arcsec)')
ax.legend()

# Dec vs time
ax = axs[1,1]
ax.errorbar(t_ast, dec_obs, yerr=dec_err, fmt='o', color='k', alpha=0.5, label='Data')
ax.plot(t_ast, Dec_best, 'r-', lw=2)
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Dec (arcsec)')
ax.legend()

# RV vs time
ax = axs[0,1]
ax.errorbar(t_rv, rv_obs, yerr=rv_err, fmt='o', color='k', alpha=0.5, label='Data')
ax.plot(t_rv, RV_best, 'r-', lw=2)
ax.set_xlabel('Time (yr)')
ax.set_ylabel('RV (km/s)')
ax.legend()

plt.tight_layout()
plt.savefig('best_fit_model.png', dpi=300)
plt.show()
