"""Spike: training-informed CS via a censored-capability delta fit.

Fits a GP adjustment delta(t) ON TOP of a frozen race-derived CS curve,
using every kept TQ point (no band selection) under an ExGaussian
(exponentially-modified Gaussian) likelihood:

    log t5k_obs_i ~ ExGaussian(mu = log t5k_race(t_i) + delta(t_i),
                               sigma = sigma_w, nu = nu_w)

i.e. observed workout time = capability + one-sided effort slack + noise.
A workout faster than the capability curve is improbable under noise alone
and pulls capability up (bounded by sigma_w — no hard ratchet); a slower
workout is attributed to slack and contributes almost nothing (exponential
tail). This is the statistical form of Max's June 2026 theory: the faster a
workout converts relative to race-derived CS, the more it can contribute.
Per Max: all TQ-accepted points enter with EQUAL weight — one global fitted
sigma_w, no per-source / per-verification distinctions.

delta has a zero-mean GP prior, so training-informed CS == race-derived CS
wherever training data is absent or all-slack. Races are never re-consumed:
the race fit stays the pure anchor (two-fit architecture — the expensive
race fit reruns per race; this fit reruns per workout sync, ~2 min).

Outputs <out>/cs_training_summary{tag}.csv with the race grid plus:
    delta_med/lo95/hi95 (log scale), cs_train_pace_med/lo95/hi95 (min/mi,
    delta CI only — race anchor held at its median), t5k_ratio columns.

One-off spike tooling (June 2026); productionize if the tests pass.
"""
import argparse
import sys
import time as tclock
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'output' / 'debug' / 'spike_cs_enrichment'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--race-summary', required=True,
                   help='bayes_cs_summary CSV of the anchoring race-only fit')
    p.add_argument('--obs', required=True,
                   help='obs CSV (date, t5k_sec, dp_fixed_m, ...) — full corpus')
    p.add_argument('--tag', default='')
    p.add_argument('--out-dir', default=str(SPIKE))
    p.add_argument('--m-basis', type=int, default=100)
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=1000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ell-mu-years', type=float, default=0.35,
                   help='LogNormal location for the delta GP lengthscale')
    p.add_argument('--likelihood',
                   choices=['emg', 'censored', 'slack-hc', 'slack-flat',
                            'quantile'],
                   default='slack-flat',
                   help="'censored' (Tobit, the theory's exact form: "
                        "P(obs|capability) = Phi((obs-mu)/sigma) — slow points "
                        "are EXACTLY flat, zero slow-side force; sigma pinned "
                        "near pair-repeatability noise) or 'emg' (ExGaussian "
                        "with fitted sigma/nu — REJECTED empirically: the "
                        "sampler explains slow mass as symmetric noise and "
                        "the slow drag returns; kept for the record)")
    p.add_argument('--race-evidence', default='',
                   help='bayes_cs_residuals CSV of the SAME race fit used as '
                        'anchor. Each kept race re-enters fit #2 as a '
                        'SYMMETRIC capability observation (sigma 0.038, the '
                        'race fit posterior sigma_base): races are the purest '
                        'capability demonstrations, so where they are dense '
                        'they pin delta (a training-informed curve refuted by '
                        'an actual race is wrong by definition); in race gaps '
                        'the workout slack-evidence rules alone.')
    p.add_argument('--sigma-race', type=float, default=0.038)
    p.add_argument('--sigma-obs', type=float, default=0.015,
                   help='censored mode: LogNormal prior center for sigma_w '
                        '(fast-side noise; default 0.015 = pair repeatability '
                        'of near-capability efforts)')
    args = p.parse_args()

    cs = pd.read_csv(args.race_summary, parse_dates=['date']).sort_values('date')
    obs = pd.read_csv(args.obs, parse_dates=['date'])

    grid_days = (cs['date'] - cs['date'].min()).dt.days.to_numpy(float)
    n_grid = len(grid_days)
    # log t5k of the race-derived curve at each grid point (capability anchor)
    log_t5k_race_grid = np.log((5000.0 - cs['dp_med'].to_numpy(float))
                               / cs['cs_mps_med'].to_numpy(float))

    obs_days = (obs['date'] - cs['date'].min()).dt.days.to_numpy(float)
    obs_idx = np.searchsorted(grid_days, obs_days)
    obs_idx = np.clip(obs_idx, 0, n_grid - 1)
    # snap to nearest grid point (searchsorted gives right neighbor)
    left = np.clip(obs_idx - 1, 0, n_grid - 1)
    use_left = (np.abs(grid_days[left] - obs_days)
                < np.abs(grid_days[obs_idx] - obs_days))
    obs_idx = np.where(use_left, left, obs_idx)

    # The obs' own anchor uses its dp_fixed (matches the projection that
    # produced t5k_sec); difference vs grid dp is negligible but keep exact.
    cs_mps_at = np.interp(obs_days, grid_days, cs['cs_mps_med'].to_numpy(float))
    log_t5k_anchor = np.log((5000.0 - obs['dp_fixed_m'].to_numpy(float)) / cs_mps_at)
    log_t5k_obs = np.log(obs['t5k_sec'].to_numpy(float))
    resid0 = log_t5k_obs - log_t5k_anchor
    print(f'{len(obs)} obs; raw log-resid vs race curve: '
          f'min {resid0.min():+.4f}, median {np.median(resid0):+.4f}, '
          f'max {resid0.max():+.4f}')

    t_c = (grid_days - grid_days.mean()) / 365.0
    L = (t_c.max() - t_c.min()) * 1.5
    X = t_c.reshape(-1, 1)

    with pm.Model():
        sf = pm.HalfNormal('sf_delta', sigma=0.05)
        ell = pm.LogNormal('ell_delta', mu=np.log(args.ell_mu_years), sigma=0.5)
        cov = sf ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ell)
        gp = pm.gp.HSGP(m=[args.m_basis], L=[L], cov_func=cov)
        delta = gp.prior('delta', X=X)

        mu_obs = log_t5k_anchor + delta[obs_idx]

        if args.race_evidence:
            rr = pd.read_csv(args.race_evidence, parse_dates=['date'])
            r_days = (rr['date'] - cs['date'].min()).dt.days.to_numpy(float)
            r_idx = np.clip(np.searchsorted(grid_days, r_days), 0, n_grid - 1)
            r_left = np.clip(r_idx - 1, 0, n_grid - 1)
            r_use_l = (np.abs(grid_days[r_left] - r_days)
                       < np.abs(grid_days[r_idx] - r_days))
            r_idx = np.where(r_use_l, r_left, r_idx)
            # The race's 5K-equivalent capability evidence relative to the
            # anchor is exactly its fit residual: log(actual/predicted).
            r_log_resid = np.log1p(rr['pct_resid'].to_numpy(float) / 100.0)
            print(f'{len(rr)} race-evidence obs from {args.race_evidence}')
            pm.Normal('obs_race', mu=delta[r_idx], sigma=args.sigma_race,
                      observed=r_log_resid)
        if args.likelihood == 'emg':
            sigma_w = pm.HalfNormal('sigma_w', sigma=0.03)
            nu_w = pm.HalfNormal('nu_w', sigma=0.10)
            pm.ExGaussian('obs_w', mu=mu_obs, sigma=sigma_w, nu=nu_w,
                          observed=log_t5k_obs)
        elif args.likelihood == 'slack-hc':
            # Proper generative censoring: obs = capability + slack + noise,
            # slack_i >= 0 latent with a HEAVY (half-Cauchy) tail. Mode at 0
            # says efforts at capability are typical; the heavy tail says a
            # soft day is unremarkable and exerts ~no pull on capability
            # (gradient ~2/r, decaying — unlike EMG's constant 1/nu, which is
            # what dragged the curve slow under hundreds of bulk points).
            # sigma_w is a hard CONSTANT: every variant that let sigma move
            # (free, or merely priored — even at 4 prior-sd of penalty)
            # collapsed into the same "wide symmetric noise, no slack"
            # attractor and the asymmetry vanished. The noise floor is an
            # ASSUMPTION (Max: "our workout-based predictions are largely
            # accurate"), pinned at near-capability pair repeatability.
            beta_s = pm.HalfNormal('beta_slack', sigma=0.05)
            slack = pm.HalfCauchy('slack', beta=beta_s, shape=len(obs))
            pm.Normal('obs_w', mu=mu_obs + slack, sigma=args.sigma_obs,
                      observed=log_t5k_obs)
        elif args.likelihood == 'slack-flat':
            # Bounded-uniform slack: slack ~ U[0, S], S beyond any real
            # point's slack. Marginal: P(obs|mu) = (Phi(r/sig) -
            # Phi((r-S)/sig))/S — a PLATEAU over the whole slack range
            # (a slow workout says NOTHING about capability: zero gradient,
            # the bulk is silent), a Gaussian edge at the fast end (frontier
            # points pull, bounded by the pinned sigma), and properly
            # normalized (no Tobit runaway: capability outrunning the
            # frontier costs likelihood). Replaces slack-hc, whose
            # mode-at-zero let the slow bulk keep attracting the curve
            # (the 2019 "slower than races" defect).
            S = 0.25
            a = (log_t5k_obs - mu_obs) / args.sigma_obs
            b = (log_t5k_obs - mu_obs - S) / args.sigma_obs
            std = pm.Normal.dist(0.0, 1.0)
            log_phi_a = pm.logcdf(std, a)
            log_phi_b = pm.logcdf(std, b)
            # log(Phi(a) - Phi(b)), stable: a - b = S/sigma >> 0 always.
            pm.Potential('obs_w', (log_phi_a
                                   + pm.math.log1mexp(log_phi_b - log_phi_a)
                                   ).sum())
        elif args.likelihood == 'quantile':
            # Frontier-quantile tracker: asymmetric Laplace pseudo-likelihood
            # with P(obs < mu) = tau = 0.10 — the curve rides the fastest
            # ~10% of local efforts. kappa^2 = tau/(1-tau).
            tau = 0.10
            kappa = float(np.sqrt(tau / (1 - tau)))
            b_q = pm.HalfNormal('b_quant', sigma=0.05)
            pm.AsymmetricLaplace('obs_w', mu=mu_obs, b=b_q, kappa=kappa,
                                 observed=log_t5k_obs)
        else:
            # Tobit / lower-bound likelihood. obs = capability + slack + noise
            # with slack >= 0 improper-flat: P(obs|mu) = Phi((obs - mu)/sigma).
            # Slow side -> 1 (flat, no force); fast side -> Gaussian penalty
            # pulling capability up. sigma_w is tightly priored at the
            # measured repeatability of near-capability efforts — the "our
            # projections are largely accurate" assumption, made explicit.
            sigma_w = pm.LogNormal('sigma_w', mu=np.log(args.sigma_obs),
                                   sigma=0.25)
            r = (log_t5k_obs - mu_obs) / sigma_w
            # pm.logcdf is the numerically stable log-Phi (no underflow on
            # the deep fast side, where the gradient matters most).
            pm.Potential('obs_w',
                         pm.logcdf(pm.Normal.dist(0.0, 1.0), r).sum())

        t0 = tclock.time()
        trace = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                          cores=min(args.chains, 8), target_accept=0.95,
                          random_seed=args.seed, return_inferencedata=True,
                          progressbar=False)
        print(f'sampled in {(tclock.time()-t0)/60:.1f} min')

    import arviz as az
    hyper = {'emg': ['sf_delta', 'ell_delta', 'sigma_w', 'nu_w'],
             'censored': ['sf_delta', 'ell_delta', 'sigma_w'],
             'slack-hc': ['sf_delta', 'ell_delta', 'beta_slack'],
             'quantile': ['sf_delta', 'ell_delta', 'b_quant']}[args.likelihood]
    summ = az.summary(trace, var_names=hyper)
    print(summ.to_string())
    n_div = int(trace.sample_stats['diverging'].sum())
    print(f'Divergences: {n_div} / {trace.sample_stats["diverging"].size}')

    d_post = trace.posterior['delta'].values.reshape(-1, n_grid)
    out = cs.copy()
    out['delta_med'] = np.median(d_post, axis=0)
    out['delta_lo95'] = np.percentile(d_post, 2.5, axis=0)
    out['delta_hi95'] = np.percentile(d_post, 97.5, axis=0)
    # Training-informed CS: t5k_train = t5k_race * exp(delta)
    # => CS_train = CS_race * exp(-delta); pace_train = pace_race * exp(delta)
    out['cs_train_mps_med'] = out['cs_mps_med'] * np.exp(-out['delta_med'])
    out['cs_train_pace_med'] = out['cs_pace_med'] * np.exp(out['delta_med'])
    out['cs_train_pace_lo95'] = out['cs_pace_med'] * np.exp(out['delta_lo95'])
    out['cs_train_pace_hi95'] = out['cs_pace_med'] * np.exp(out['delta_hi95'])
    suffix = f'_{args.tag}' if args.tag else ''
    path = Path(args.out_dir) / f'cs_training_summary{suffix}.csv'
    out.to_csv(path, index=False)
    print(f'Wrote {path}')

    dmin_i = int(np.argmin(out['delta_med'].to_numpy()))
    print(f"max upward pull: delta {out['delta_med'].iloc[dmin_i]:+.4f} "
          f"({out['delta_med'].iloc[dmin_i]*330:+.1f} s/mi-ish) "
          f"on {out['date'].iloc[dmin_i].date()}")


if __name__ == '__main__':
    main()
