"""
Physical MOND orbit model for fitting S2's orbit around Sgr A*.

UNIT CONVENTIONS (fixed, documented once, used everywhere):
    theta = [a, e, i_deg, omega_deg, Omega_deg, T0, M, a0]
        a      : semi-major axis, in units of 1e3 AU      (code range ~0.3-3.0)
        e      : eccentricity                              (0-1)
        i_deg  : inclination, degrees                       (0-180)
        omega_deg : argument of periapsis, degrees           (0-360)
        Omega_deg : longitude of ascending node, degrees     (0-360)
        T0     : time of pericenter passage, in the SAME time units as
                 the data's "t" column (years - whatever epoch convention
                 the data uses, e.g. calendar year)
        M      : Sgr A* mass, in units of 1e6 Msun          (code range ~1-8)
        a0     : MOND acceleration constant, in AU/yr^2 (see note below)

    Observables:
        RA_model, Dec_model : arcsec
        RV_model             : km/s
        t                    : years (same epoch convention as T0)

Internally the orbit is integrated in physical AU and AU/yr using the
correct Kepler constant (G = 4*pi^2 in AU-yr-Msun units), then converted
back to arcsec (via the assumed distance R0 to Sgr A*) and km/s (via the
exact AU/yr -> km/s conversion factor) for comparison to data.
"""

import numpy as np
from scipy.integrate import solve_ivp

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


def orbit_model_mond(t, theta, n_ast, rtol=1e-10, atol=1e-12):
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
    rtol, atol : float
        solve_ivp tolerances. Defaults are tight (matches the precision
        needed since the MOND signal in S2 is a tiny secular effect).
        Pass looser values (e.g. rtol=1e-8, atol=1e-9) for a fast global
        search where you don't yet need per-mille precision - just don't
        use loose tolerances for the final production MCMC run.

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

    # IMPORTANT: solve_ivp always applies y0 at t_span[0], not at t=0. The
    # pericenter state in y0 is only correct if we integrate starting FROM
    # t_rel=0 - otherwise T0 cancels out of the calculation entirely (this
    # was the actual bug: t_span used to be built from t_rel.min()/.max(),
    # which silently anchored pericenter to the start of the data window
    # regardless of T0). Fix: integrate forward (0 -> max) and backward
    # (0 -> min) separately, both starting from the true pericenter state.
    n_unique = len(unique_t_rel)
    x_orb = np.zeros(n_unique)
    y_orb = np.zeros(n_unique)
    vx_orb = np.zeros(n_unique)
    vy_orb = np.zeros(n_unique)

    pos_mask = unique_t_rel >= 0
    neg_mask = ~pos_mask

    try:
        if np.any(pos_mask):
            t_pos = unique_t_rel[pos_mask]  # ascending, >= 0
            sol_f = solve_ivp(
                mond_orbit_equations, (0.0, t_pos.max() + 0.1), y0, args=(GM, a0),
                t_eval=t_pos, method='DOP853', rtol=rtol, atol=atol,
            )
            if not sol_f.success:
                return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)
            x_orb[pos_mask] = sol_f.y[0]
            y_orb[pos_mask] = sol_f.y[1]
            vx_orb[pos_mask] = sol_f.y[2]
            vy_orb[pos_mask] = sol_f.y[3]

        if np.any(neg_mask):
            t_neg = unique_t_rel[neg_mask]           # ascending, all < 0
            t_neg_desc = t_neg[::-1]                  # descending: near-zero -> most negative
            sol_b = solve_ivp(
                mond_orbit_equations, (0.0, t_neg.min() - 0.1), y0, args=(GM, a0),
                t_eval=t_neg_desc, method='DOP853', rtol=rtol, atol=atol,
            )
            if not sol_b.success:
                return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)
            # sol_b columns are in descending-time order; flip back to match
            # the ascending order of t_neg / neg_mask.
            x_orb[neg_mask] = sol_b.y[0][::-1]
            y_orb[neg_mask] = sol_b.y[1][::-1]
            vx_orb[neg_mask] = sol_b.y[2][::-1]
            vy_orb[neg_mask] = sol_b.y[3][::-1]
    except Exception:
        return np.zeros(n_ast), np.zeros(n_ast), np.zeros(n - n_ast)

    x_orb = x_orb[inv_idx]
    y_orb = y_orb[inv_idx]
    vx_orb = vx_orb[inv_idx]
    vy_orb = vy_orb[inv_idx]

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
