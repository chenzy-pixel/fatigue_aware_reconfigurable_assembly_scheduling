"""Paired GAE diagnostics on complete, frozen-policy trajectories."""
from __future__ import annotations

from itertools import combinations
import numpy as np

LAMBDAS = (0.9, 0.95, 0.97, 0.99, 1.0)


def estimates(rewards, values, *, gamma=1.0, lambdas=LAMBDAS):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or values.shape != rewards.shape:
        raise ValueError("a nonempty complete episode requires aligned rewards and values")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("rewards and values must be finite")
    if not 0 <= gamma <= 1 or any(not 0 <= x <= 1 for x in lambdas):
        raise ValueError("gamma and lambda must be in [0, 1]")
    mc = np.empty_like(rewards)
    tail = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        tail = rewards[t] + gamma * tail
        mc[t] = tail
    delta = rewards + gamma * np.r_[values[1:], 0.0] - values
    advantages = {}
    for lam in lambdas:
        a = np.empty_like(rewards)
        tail = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            tail = delta[t] + gamma * lam * tail
            a[t] = tail
        advantages[lam] = a
    if 1.0 in advantages:
        np.testing.assert_allclose(advantages[1.0], mc - values, atol=1e-10, rtol=1e-10)
    return advantages, mc


def sign_statistics(a, b, *, epsilon=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    valid = (np.abs(a) > epsilon) & (np.abs(b) > epsilon)
    opposite = (a * b < 0) & valid
    return {
        "sign_agreement": float((np.sign(a) == np.sign(b)).mean()),
        "nonzero_sign_agreement": float((~opposite[valid]).mean()) if valid.any() else None,
        "opposite_fraction": float(opposite.mean()),
        "nonzero_steps": int(valid.sum()),
        "near_zero_fraction": float((~valid).mean()),
    }


def aggregate(episodes):
    """Report step-weighted and episode-weighted metrics; normalize once per role.

    Diagnostic centering uses the fixed evaluation batch, not a reconstructed
    historical training minibatch. Choice-only rows exclude zero policy-gradient
    states with exactly one legal action, while GAE includes their rewards/values.
    """
    if not episodes:
        raise ValueError("no complete episodes")
    rows, pairs, trajectory_rows = [], [], []
    roles = sorted({e["role"] for e in episodes})
    for role in roles:
        group = [e for e in episodes if e["role"] == role]
        for e in group:
            e["advantages"], e["mc"] = estimates(e["rewards"], e["values"], gamma=e["gamma"])
            for lam in LAMBDAS:
                a = e["advantages"][lam]
                error = a + e["values"] - e["mc"]
                trajectory_rows.append({**e["metadata"], "lambda": lam,
                    "steps": len(a), "choice_steps": int((e["legal_counts"] > 1).sum()),
                    "advantage_variance": float(a.var()), "bias": float(error.mean()),
                    "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
                    **sign_statistics(a, e["advantages"][1.0])})
        centers = {lam: np.concatenate([e["advantages"][lam] for e in group]).mean() for lam in LAMBDAS}
        for region in ("all", "flow", "cost", "variance", "center"):
            for outcome in ("all", "success", "failure"):
                selected = [e for e in group if (region == "all" or e["region"] == region)
                            and (outcome == "all" or e["success"] == (outcome == "success"))]
                if not selected:
                    continue
                for scope in ("all_steps", "choice_steps"):
                    masks = [np.ones(len(e["values"]), dtype=bool) if scope == "all_steps"
                             else e["legal_counts"] > 1 for e in selected]
                    if not sum(int(m.sum()) for m in masks):
                        continue
                    joined = lambda key: np.concatenate([e[key][m] for e, m in zip(selected, masks)])
                    mc, v = joined("mc"), joined("values")
                    adv = {lam: np.concatenate([e["advantages"][lam][m] for e, m in zip(selected, masks)])
                           for lam in LAMBDAS}
                    base = {"role": role, "region": region, "outcome": outcome, "scope": scope,
                            "trajectories": len(selected), "instances": len({e["metadata"]["instance_id"] for e in selected}),
                            "steps": len(v)}
                    for lam, a in adv.items():
                        error = a + v - mc
                        per_episode_bias = [float((e["advantages"][lam] + e["values"] - e["mc"])[m].mean())
                                            for e, m in zip(selected, masks) if m.any()]
                        rows.append({**base, "lambda": lam, "advantage_variance": float(a.var()),
                            "advantage_mean": float(a.mean()), "mc_return_variance": float(mc.var()),
                            "bias": float(error.mean()), "mae": float(np.abs(error).mean()),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "episode_weighted_bias": float(np.mean(per_episode_bias)),
                            "critic_mc_bias": float((v - mc).mean()),
                            "critic_mc_explained_variance": float(1 - (mc - v).var()/mc.var()) if mc.var() > 1e-12 else None,
                            "positive_fraction": float((a > 0).mean())})
                    for left, right in combinations(LAMBDAS, 2):
                        raw = sign_statistics(adv[left], adv[right])
                        centered = sign_statistics(adv[left] - centers[left], adv[right] - centers[right])
                        pairs.append({**base, "lambda_left": left, "lambda_right": right, **raw,
                                      **{"centered_" + k: val for k, val in centered.items()}})
    return rows, pairs, trajectory_rows


def diagnostic_gate(rows, pairs):
    """Predeclared practical screen; these thresholds are not a statistical theorem."""
    failures = []
    for row in rows:
        if row["lambda"] != 0.95 or row["scope"] != "choice_steps" or row["outcome"] == "success":
            continue
        pair = next(p for p in pairs if all(p[k] == row[k] for k in ("role", "region", "outcome", "scope"))
                    and p["lambda_left"] == 0.95 and p["lambda_right"] == 1.0)
        if row["rmse"] > 0.05 or pair["nonzero_sign_agreement"] < 0.95 or pair["centered_nonzero_sign_agreement"] < 0.95:
            failures.append({k: row[k] for k in ("role", "region", "outcome", "trajectories", "instances", "rmse")}
                            | {"sign_agreement": pair["nonzero_sign_agreement"],
                               "centered_sign_agreement": pair["centered_nonzero_sign_agreement"]})
    return {"requires_training": bool(failures), "minimum_sign_agreement": 0.95,
            "maximum_return_rmse": 0.05, "failures": failures,
            "interpretation": "Paired estimator differences on frozen policies; not proof of the true policy-gradient direction or a winning lambda."}
