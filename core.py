import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MatchModel:
    home: str
    away: str
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(lh: float, la: float, rho: float = -0.05, max_goals: int = 8) -> np.ndarray:
    m = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i, j] = poisson_pmf(i, lh) * poisson_pmf(j, la)
    # Dixon-Coles low-score adjustment.
    adjustments = {
        (0, 0): 1 - lh * la * rho,
        (0, 1): 1 + lh * rho,
        (1, 0): 1 + la * rho,
        (1, 1): 1 - rho,
    }
    for (i, j), tau in adjustments.items():
        m[i, j] *= max(0.0, tau)
    total = m.sum()
    return m / total if total else m


def team_strengths(hist: pd.DataFrame, as_of: Optional[pd.Timestamp] = None,
                   half_life: int = 180, shrink_k: float = 5.0):
    if hist is None or hist.empty:
        raise ValueError("Historique vide")
    hist = hist.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    if hist.empty:
        raise ValueError("Historique inutilisable")
    as_of = pd.Timestamp(as_of) if as_of is not None else hist["Date"].max() + pd.Timedelta(days=1)
    age = (as_of - hist["Date"]).dt.days.clip(lower=0)
    w = np.exp(-np.log(2) * age / max(half_life, 1))
    lg_h = float(np.average(hist["FTHG"].astype(float), weights=w))
    lg_a = float(np.average(hist["FTAG"].astype(float), weights=w))
    stats = {}
    teams = sorted(set(hist["HomeTeam"]) | set(hist["AwayTeam"]))
    for team in teams:
        h = hist[hist["HomeTeam"] == team]
        a = hist[hist["AwayTeam"] == team]
        wh = w.loc[h.index]
        wa = w.loc[a.index]
        if len(h):
            hf, hc = np.average(h["FTHG"], weights=wh), np.average(h["FTAG"], weights=wh)
        else:
            hf, hc = lg_h, lg_a
        if len(a):
            af, ac = np.average(a["FTAG"], weights=wa), np.average(a["FTHG"], weights=wa)
        else:
            af, ac = lg_a, lg_h
        hf = (hf * wh.sum() + lg_h * shrink_k) / (wh.sum() + shrink_k)
        hc = (hc * wh.sum() + lg_a * shrink_k) / (wh.sum() + shrink_k)
        af = (af * wa.sum() + lg_a * shrink_k) / (wa.sum() + shrink_k)
        ac = (ac * wa.sum() + lg_h * shrink_k) / (wa.sum() + shrink_k)
        stats[team] = {
            "ha": hf / max(lg_h, 0.1),
            "hd": hc / max(lg_a, 0.1),
            "aa": af / max(lg_a, 0.1),
            "ad": ac / max(lg_h, 0.1),
            "matches": int(len(h) + len(a)),
        }
    return stats, lg_h, lg_a


def calculate_elo(hist: pd.DataFrame, k: float = 20, home_adv: float = 55) -> Dict[str, float]:
    elo: Dict[str, float] = {}
    for _, r in hist.sort_values("Date").iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        elo.setdefault(h, 1500.0)
        elo.setdefault(a, 1500.0)
        eh = 1 / (1 + 10 ** (-((elo[h] + home_adv - elo[a]) / 400)))
        res = 1.0 if r["FTHG"] > r["FTAG"] else 0.5 if r["FTHG"] == r["FTAG"] else 0.0
        elo[h] += k * (res - eh)
        elo[a] += k * ((1 - res) - (1 - eh))
    return elo


def fit_goal_model(hist: pd.DataFrame, home: str, away: str) -> MatchModel:
    strengths, lg_h, lg_a = team_strengths(hist)
    elo = calculate_elo(hist)
    hs = strengths.get(home)
    aws = strengths.get(away)
    # For a team outside the historical base, fall back to league-average strength.
    hs = hs or {"ha": 1.0, "hd": 1.0, "aa": 1.0, "ad": 1.0, "matches": 0}
    aws = aws or {"ha": 1.0, "hd": 1.0, "aa": 1.0, "ad": 1.0, "matches": 0}
    lh = float(np.clip(lg_h * hs["ha"] * aws["ad"], 0.15, 4.5))
    la = float(np.clip(lg_a * aws["aa"] * hs["hd"], 0.15, 4.5))
    eh = elo.get(home, 1500) + 55
    ea = elo.get(away, 1500)
    elo_home = 1 / (1 + 10 ** (-((eh - ea) / 400)))
    # Small Elo correction, deliberately capped to avoid overpowering goals model.
    delta = (elo_home - 0.5) * 0.20
    lh *= float(np.clip(1 + delta, 0.90, 1.10))
    la *= float(np.clip(1 - delta, 0.90, 1.10))
    return MatchModel(home, away, lh, la, score_matrix(lh, la))


def probabilities(model: MatchModel) -> Dict[str, float]:
    m = model.matrix
    p_home = float(np.triu(m, 1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.tril(m, -1).sum())
    total_goals = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    p = {
        "1": p_home, "X": p_draw, "2": p_away,
        "1X": p_home + p_draw, "X2": p_draw + p_away, "12": p_home + p_away,
        "DNB 1": p_home / max(1 - p_draw, 1e-9),
        "DNB 2": p_away / max(1 - p_draw, 1e-9),
    }
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under = float(m[total_goals <= line].sum())
        p[f"Over {line:g}"] = 1 - under
        p[f"Under {line:g}"] = under
    btts_yes = float(m[1:, 1:].sum())
    p["BTTS Oui"] = btts_yes
    p["BTTS Non"] = 1 - btts_yes
    p["Domicile marque"] = 1 - float(m[0, :].sum())
    p["Extérieur marque"] = 1 - float(m[:, 0].sum())
    p["Domicile clean sheet"] = float(m[:, 0].sum())
    p["Extérieur clean sheet"] = float(m[0, :].sum())
    p["Domicile gagne sans encaisser"] = float(m[1:, 0].sum())
    p["Extérieur gagne sans encaisser"] = float(m[0, 1:].sum())
    p["Aucun but"] = float(m[0, 0])
    p["Domicile marque en premier"] = (1 - float(m[0, 0])) * model.lambda_home / max(model.lambda_home + model.lambda_away, 1e-9)
    p["Extérieur marque en premier"] = (1 - float(m[0, 0])) * model.lambda_away / max(model.lambda_home + model.lambda_away, 1e-9)
    # Exact scores: useful as ranking, but intentionally not treated as a low-risk market.
    flat = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            flat.append((float(m[i, j]), f"{model.home} {i}-{j} {model.away}"))
    for prob, label in sorted(flat, reverse=True)[:5]:
        p[f"Score exact {label}"] = prob
    return {k: float(np.clip(v, 0, 1)) for k, v in p.items()}


def fair_odds(prob: float) -> Optional[float]:
    return None if prob <= 0 else 1 / prob


def edge(prob: float, decimal_odds: float) -> float:
    return prob * decimal_odds - 1


def integrity_proxy(probs: Dict[str, float], market_probs: Optional[Dict[str, float]], data_quality: float = 1.0) -> Tuple[int, List[str]]:
    """Not a match-fixing detector. Scores unusual disagreement / data weakness only."""
    score = 0
    reasons: List[str] = []
    if data_quality < 0.6:
        score += 20
        reasons.append("Données historiques limitées")
    if market_probs:
        diffs = [abs(probs[k] - market_probs[k]) for k in probs.keys() if k in market_probs]
        if diffs:
            mx = max(diffs)
            if mx > 0.20:
                score += 45; reasons.append("Écart modèle/marché très élevé")
            elif mx > 0.12:
                score += 25; reasons.append("Écart modèle/marché inhabituel")
            elif mx > 0.08:
                score += 10; reasons.append("Écart modèle/marché notable")
    if not reasons:
        reasons.append("Aucune anomalie forte dans les signaux disponibles")
    return min(score, 100), reasons


def market_candidates(model: MatchModel, odds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    probs = probabilities(model)
    rows = []
    for market, prob in probs.items():
        if market.startswith("Score exact"):
            risk = "très élevé"
        elif prob >= 0.78:
            risk = "faible"
        elif prob >= 0.65:
            risk = "modéré"
        else:
            risk = "élevé"
        odd = odds.get(market) if odds else None
        ev = edge(prob, odd) if odd and odd > 1 else None
        rows.append({"Marché": market, "Probabilité": prob, "Cote": odd, "EV": ev, "Risque": risk})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Score"] = df["Probabilité"] + df["EV"].fillna(0).clip(-0.25, 0.25) * 0.25
        df = df.sort_values(["Score", "Probabilité"], ascending=False)
    return df.reset_index(drop=True)


def choose_best_market(model: MatchModel, odds: Dict[str, float], min_prob: float = 0.60) -> Optional[dict]:
    df = market_candidates(model, odds)
    if df.empty:
        return None
    df = df[(df["Probabilité"] >= min_prob) & df["Cote"].notna() & (df["Cote"] > 1)]
    if df.empty:
        return None
    # Prefer positive EV, then probability, while penalising exact-score markets.
    df = df.sort_values(["EV", "Probabilité"], ascending=False)
    row = df.iloc[0]
    if float(row["EV"]) < 0:
        row = df.sort_values("Probabilité", ascending=False).iloc[0]
    return row.to_dict()


def build_combo(selections: List[dict], profile: str) -> dict:
    targets = {"Prudent": (0.72, 0.55), "Équilibré": (0.63, 0.65), "Grosse cote": (0.48, 0.90)}
    min_prob, _ = targets.get(profile, targets["Équilibré"])
    # Different matches are treated as approximately independent for a first version.
    chosen = [s for s in selections if s.get("prob", 0) >= min_prob and s.get("odds", 0) > 1]
    max_n = 5 if profile == "Prudent" else 7 if profile == "Équilibré" else 10
    chosen = sorted(chosen, key=lambda x: (x.get("ev", -9), x.get("prob", 0)), reverse=True)[:max_n]
    if not chosen:
        return {"selections": [], "odds": 1.0, "prob": 0.0, "note": "Aucune sélection ne respecte le filtre."}
    total_odds = float(np.prod([x["odds"] for x in chosen]))
    joint_prob = float(np.prod([x["prob"] for x in chosen]))
    return {"selections": chosen, "odds": total_odds, "prob": joint_prob,
            "note": "Probabilité combinée approximative, car les matchs sont supposés indépendants."}
