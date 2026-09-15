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
    adjustments = {(0, 0): 1 - lh * la * rho, (0, 1): 1 + lh * rho,
                   (1, 0): 1 + la * rho, (1, 1): 1 - rho}
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
        wh, wa = w.loc[h.index], w.loc[a.index]
        hf, hc = (np.average(h["FTHG"], weights=wh), np.average(h["FTAG"], weights=wh)) if len(h) else (lg_h, lg_a)
        af, ac = (np.average(a["FTAG"], weights=wa), np.average(a["FTHG"], weights=wa)) if len(a) else (lg_a, lg_h)
        hf = (hf * wh.sum() + lg_h * shrink_k) / (wh.sum() + shrink_k)
        hc = (hc * wh.sum() + lg_a * shrink_k) / (wh.sum() + shrink_k)
        af = (af * wa.sum() + lg_a * shrink_k) / (wa.sum() + shrink_k)
        ac = (ac * wa.sum() + lg_h * shrink_k) / (wa.sum() + shrink_k)
        stats[team] = {"ha": hf / max(lg_h, 0.1), "hd": hc / max(lg_a, 0.1),
                       "aa": af / max(lg_a, 0.1), "ad": ac / max(lg_h, 0.1),
                       "matches": int(len(h) + len(a))}
    return stats, lg_h, lg_a


def calculate_elo(hist: pd.DataFrame, k: float = 20, home_adv: float = 55) -> Dict[str, float]:
    elo: Dict[str, float] = {}
    for _, r in hist.sort_values("Date").iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        elo.setdefault(h, 1500.0); elo.setdefault(a, 1500.0)
        eh = 1 / (1 + 10 ** (-((elo[h] + home_adv - elo[a]) / 400)))
        res = 1.0 if r["FTHG"] > r["FTAG"] else 0.5 if r["FTHG"] == r["FTAG"] else 0.0
        elo[h] += k * (res - eh); elo[a] += k * ((1 - res) - (1 - eh))
    return elo


def fit_goal_model(hist: pd.DataFrame, home: str, away: str) -> MatchModel:
    strengths, lg_h, lg_a = team_strengths(hist)
    elo = calculate_elo(hist)
    hs = strengths.get(home) or {"ha": 1.0, "hd": 1.0, "aa": 1.0, "ad": 1.0, "matches": 0}
    aws = strengths.get(away) or {"ha": 1.0, "hd": 1.0, "aa": 1.0, "ad": 1.0, "matches": 0}
    lh = float(np.clip(lg_h * hs["ha"] * aws["ad"], 0.15, 4.5))
    la = float(np.clip(lg_a * aws["aa"] * hs["hd"], 0.15, 4.5))
    eh, ea = elo.get(home, 1500) + 55, elo.get(away, 1500)
    elo_home = 1 / (1 + 10 ** (-((eh - ea) / 400)))
    delta = (elo_home - 0.5) * 0.20
    lh *= float(np.clip(1 + delta, 0.90, 1.10)); la *= float(np.clip(1 - delta, 0.90, 1.10))
    return MatchModel(home, away, lh, la, score_matrix(lh, la))


def probabilities(model: MatchModel) -> Dict[str, float]:
    m = model.matrix
    p_home, p_draw, p_away = float(np.triu(m, 1).sum()), float(np.trace(m)), float(np.tril(m, -1).sum())
    total_goals = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    p = {"1": p_home, "X": p_draw, "2": p_away,
         "1X": p_home + p_draw, "X2": p_draw + p_away, "12": p_home + p_away,
         "DNB 1": p_home / max(1 - p_draw, 1e-9), "DNB 2": p_away / max(1 - p_draw, 1e-9)}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under = float(m[total_goals <= line].sum())
        p[f"Over {line:g}"] = 1 - under; p[f"Under {line:g}"] = under
    btts_yes = float(m[1:, 1:].sum())
    p["BTTS Oui"], p["BTTS Non"] = btts_yes, 1 - btts_yes
    p["Domicile marque"], p["Extérieur marque"] = 1 - float(m[0, :].sum()), 1 - float(m[:, 0].sum())
    p["Domicile clean sheet"], p["Extérieur clean sheet"] = float(m[:, 0].sum()), float(m[0, :].sum())
    p["Domicile gagne sans encaisser"], p["Extérieur gagne sans encaisser"] = float(m[1:, 0].sum()), float(m[0, 1:].sum())
    p["Aucun but"] = float(m[0, 0])
    nonzero = 1 - float(m[0, 0])
    p["Domicile marque en premier"] = nonzero * model.lambda_home / max(model.lambda_home + model.lambda_away, 1e-9)
    p["Extérieur marque en premier"] = nonzero * model.lambda_away / max(model.lambda_home + model.lambda_away, 1e-9)
    flat = [(float(m[i, j]), f"{model.home} {i}-{j} {model.away}") for i in range(m.shape[0]) for j in range(m.shape[1])]
    for prob, label in sorted(flat, reverse=True)[:5]: p[f"Score exact {label}"] = prob
    return {k: float(np.clip(v, 0, 1)) for k, v in p.items()}


def fair_odds(prob: float) -> Optional[float]:
    return None if prob <= 0 else 1 / prob


def edge(prob: float, decimal_odds: float) -> float:
    return prob * decimal_odds - 1


def normalize_1x2_odds(odds: Dict[str, float]) -> Dict[str, float]:
    vals = {k: odds.get(k) for k in ("1", "X", "2")}
    if any(v is None or v <= 1 for v in vals.values()): return {}
    inv = np.array([1 / vals["1"], 1 / vals["X"], 1 / vals["2"]], dtype=float)
    inv /= inv.sum()
    return {"1": float(inv[0]), "X": float(inv[1]), "2": float(inv[2])}


def market_candidates(model: MatchModel, odds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    probs = probabilities(model); rows = []
    for market, prob in probs.items():
        risk = "très élevé" if market.startswith("Score exact") else "faible" if prob >= 0.78 else "modéré" if prob >= 0.65 else "élevé"
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
    if df.empty: return None
    df = df[(df["Probabilité"] >= min_prob) & df["Cote"].notna() & (df["Cote"] > 1)]
    if df.empty: return None
    df = df.sort_values(["EV", "Probabilité"], ascending=False)
    row = df.iloc[0]
    if float(row["EV"]) < 0: row = df.sort_values("Probabilité", ascending=False).iloc[0]
    return row.to_dict()


def bookmaker_dispersion(bookmaker_odds: List[Dict[str, float]]) -> Tuple[float, List[str]]:
    """Returns 0-25 market-dispersion score and explanations. It is not proof of fixing."""
    probs = [normalize_1x2_odds(x) for x in bookmaker_odds]
    probs = [p for p in probs if p]
    if len(probs) < 2: return 0.0, ["Moins de 2 bookmakers exploitables"]
    arr = np.array([[p["1"], p["X"], p["2"]] for p in probs])
    spread = float(np.max(arr, axis=0).max() - np.min(arr, axis=0).min())
    if spread >= 0.15: return 25.0, ["Dispersion très forte entre bookmakers"]
    if spread >= 0.10: return 18.0, ["Dispersion inhabituelle entre bookmakers"]
    if spread >= 0.06: return 10.0, ["Dispersion notable entre bookmakers"]
    return 0.0, ["Marché relativement cohérent entre bookmakers"]


def odds_movement_score(previous: Optional[Dict[str, float]], current: Dict[str, float]) -> Tuple[float, str]:
    if not previous or not current: return 0.0, "Pas d'historique de mouvement disponible"
    changes = []
    for k in ("1", "X", "2", "Over 2.5", "Under 2.5", "BTTS Oui", "BTTS Non"):
        a, b = previous.get(k), current.get(k)
        if a and b and a > 1 and b > 1:
            changes.append(abs(b / a - 1))
    if not changes: return 0.0, "Pas assez de cotes comparables"
    mx = max(changes)
    if mx >= 0.20: return 25.0, f"Mouvement de cote très brutal ({mx*100:.0f}%)"
    if mx >= 0.12: return 18.0, f"Mouvement de cote important ({mx*100:.0f}%)"
    if mx >= 0.08: return 10.0, f"Mouvement de cote notable ({mx*100:.0f}%)"
    return 0.0, "Mouvement de cote modéré"


def league_risk_score(league: str) -> Tuple[float, str]:
    s = (league or "").lower()
    high = ["friendly", "amical", "u19", "u20", "u21", "u23", "reserve", "reserves", "youth", "regional"]
    medium = ["2.", "3.", "4.", "5.", "lower", "national", "division 2", "division 3"]
    if any(x in s for x in high): return 10.0, "Compétition à faible visibilité / amicale / jeunes"
    if any(x in s for x in medium): return 5.0, "Compétition moins exposée"
    return 0.0, "Pas de signal de risque lié au type de compétition"


def integrity_assessment(model_probs: Dict[str, float], market_probs: Optional[Dict[str, float]],
                         bookmaker_odds: Optional[List[Dict[str, float]]] = None,
                         previous_odds: Optional[Dict[str, float]] = None,
                         league: str = "", data_quality: float = 1.0,
                         performance_signal: float = 0.0) -> Tuple[int, List[str], Dict[str, float]]:
    """Market-surveillance score, not a match-fixing detector."""
    score, reasons = 0.0, []
    if market_probs:
        diffs = [abs(model_probs[k] - market_probs[k]) for k in ("1", "X", "2") if k in model_probs and k in market_probs]
        if diffs:
            mx = max(diffs)
            add = 35 if mx > .20 else 25 if mx > .12 else 12 if mx > .08 else 0
            score += add
            if add: reasons.append(f"Écart modèle/marché élevé ({mx*100:.0f} points")
    if bookmaker_odds:
        add, rs = bookmaker_dispersion(bookmaker_odds); score += add; reasons.extend(rs if add else [])
    move, move_reason = odds_movement_score(previous_odds, bookmaker_odds[0] if bookmaker_odds else {})
    score += move
    if move: reasons.append(move_reason)
    lr, lr_reason = league_risk_score(league); score += lr
    if lr: reasons.append(lr_reason)
    if performance_signal > 0:
        score += min(15.0, performance_signal); reasons.append("Signal statistique en direct inhabituel")
    if data_quality < 0.45:
        reasons.append("Données insuffisantes: score à interpréter avec prudence")
    if not reasons: reasons.append("Aucun signal de marché inhabituel détecté")
    score = int(round(min(100, score)))
    return score, reasons, {"model_market": float(score), "market_dispersion": 0.0, "odds_movement": move, "league_risk": lr}


def classify_integrity(score: int) -> str:
    if score >= 70: return "🔴 Surveillance élevée"
    if score >= 50: return "🟠 Anomalie notable"
    if score >= 30: return "🟡 À surveiller"
    return "🟢 Aucun signal fort"


def integrity_proxy(probs: Dict[str, float], market_probs: Optional[Dict[str, float]], data_quality: float = 1.0):
    score, reasons, _ = integrity_assessment(probs, market_probs, data_quality=data_quality)
    return score, reasons


def build_combo(selections: List[dict], profile: str) -> dict:
    targets = {"Prudent": (0.72, 0.55), "Équilibré": (0.63, 0.65), "Grosse cote": (0.48, 0.90)}
    min_prob, _ = targets.get(profile, targets["Équilibré"])
    chosen = [s for s in selections if s.get("prob", 0) >= min_prob and s.get("odds", 0) > 1 and s.get("integrity", 0) < 50]
    max_n = 5 if profile == "Prudent" else 7 if profile == "Équilibré" else 10
    chosen = sorted(chosen, key=lambda x: (x.get("ev", -9), x.get("prob", 0)), reverse=True)[:max_n]
    if not chosen: return {"selections": [], "odds": 1.0, "prob": 0.0, "note": "Aucune sélection suffisamment solide et sans anomalie notable."}
    total_odds = float(np.prod([x["odds"] for x in chosen])); joint_prob = float(np.prod([x["prob"] for x in chosen]))
    return {"selections": chosen, "odds": total_odds, "prob": joint_prob,
            "note": "Probabilité combinée approximative: les matchs sont supposés indépendants."}

# ---------------- CharioBet Decision Engine v0.7 ----------------
def devig_market_consensus(bookmakers: List[Dict[str, float]], markets: Optional[List[str]] = None) -> Dict[str, float]:
    """Consensus probability across available bookmakers, removing each book's overround market by market."""
    if not bookmakers:
        return {}
    markets = markets or ["1", "X", "2"]
    acc = {m: [] for m in markets}
    for b in bookmakers:
        vals = {m: b.get(m) for m in markets}
        valid = {m: v for m, v in vals.items() if v and v > 1}
        if len(valid) < 2:
            continue
        inv = {m: 1.0 / v for m, v in valid.items()}
        s = sum(inv.values())
        if not s:
            continue
        for m in valid:
            acc[m].append(inv[m] / s)
    return {m: float(np.mean(v)) for m, v in acc.items() if v}


def market_anchored_prob(model_prob: float, market_prob: Optional[float],
                         model_weight: float = 0.65) -> float:
    """Conservative ensemble: model + market consensus. Never treats either source as infallible."""
    if market_prob is None:
        return float(np.clip(model_prob, 0.01, 0.99))
    w = float(np.clip(model_weight, 0.35, 0.85))
    return float(np.clip(w * model_prob + (1 - w) * market_prob, 0.01, 0.99))


def decision_grade(prob: float, edge_value: Optional[float], confidence: float = 1.0,
                   integrity: int = 0, data_quality: float = 1.0) -> Dict[str, object]:
    """Strict gate inspired by transparent value-model platforms: low quality => no forced pick."""
    p = float(prob); conf = float(confidence); dq = float(data_quality)
    ev = float(edge_value) if edge_value is not None else None
    score = 100 * (0.55 * p + 0.25 * conf + 0.20 * dq)
    if ev is not None:
        score += float(np.clip(ev, -0.20, 0.20)) * 60
    score -= max(0, integrity - 29) * 0.70
    score = float(np.clip(score, 0, 100))
    if integrity >= 50:
        verdict = "NO BET — intégrité à surveiller"
    elif conf < 0.65 or dq < 0.50:
        verdict = "NO BET — données insuffisantes"
    elif p < 0.60:
        verdict = "NO BET — probabilité trop faible"
    elif ev is not None and ev < 0.02:
        verdict = "SURVEILLER — valeur faible"
    else:
        verdict = "SIGNAL VALIDE"
    return {"score": round(score, 1), "verdict": verdict}


def select_consensus_market(model: MatchModel, bookmakers: Optional[List[Dict[str, float]]] = None,
                            min_prob: float = 0.60, min_edge: float = 0.02,
                            integrity: int = 0, data_quality: float = 1.0) -> Optional[dict]:
    """Select only markets with sufficient probability/value; otherwise explicitly return no bet."""
    probs = probabilities(model)
    consensus = devig_market_consensus(bookmakers or [])
    rows = []
    for market, p in probs.items():
        mp = consensus.get(market)
        fair = 1 / p if p > 0 else None
        # Use consensus only as an anchor. If no market consensus exists, do not invent value.
        ev = (p / mp - 1) if mp and mp > 0 else None
        if p < min_prob or ev is None or ev < min_edge:
            continue
        grade = decision_grade(p, ev, confidence=min(1.0, data_quality), integrity=integrity, data_quality=data_quality)
        if grade["verdict"] == "SIGNAL VALIDE":
            rows.append({"Marché": market, "Probabilité": p, "Prob marché": mp,
                         "Cote juste": fair, "Edge": ev, "Score décision": grade["score"],
                         "Verdict": grade["verdict"]})
    if not rows:
        return None
    return sorted(rows, key=lambda x: (x["Score décision"], x["Edge"], x["Probabilité"]), reverse=True)[0]


def combo_risk_adjusted(selections: List[dict], profile: str) -> dict:
    """Build a multi-match combo while penalising weak legs and excessive concentration."""
    targets = {
        "Prudent": {"min_prob": .72, "max_n": 4, "min_edge": .01},
        "Équilibré": {"min_prob": .64, "max_n": 6, "min_edge": .02},
        "Grosse cote": {"min_prob": .52, "max_n": 8, "min_edge": .04},
    }
    cfg = targets.get(profile, targets["Équilibré"])
    pool = [s for s in selections if s.get("integrity", 0) < 50 and s.get("prob", 0) >= cfg["min_prob"]
            and s.get("odds", 0) > 1 and s.get("ev", -1) >= cfg["min_edge"]]
    # Different match is already enforced by the caller; avoid stacking identical market families.
    chosen = []
    families = set()
    for s in sorted(pool, key=lambda x: (x.get("decision_score", 0), x.get("ev", -1), x.get("prob", 0)), reverse=True):
        fam = str(s.get("market", "")).split(" ")[0]
        if fam in families and profile != "Grosse cote":
            continue
        chosen.append(s); families.add(fam)
        if len(chosen) >= cfg["max_n"]: break
    if not chosen:
        return {"selections": [], "odds": 1.0, "prob": 0.0,
                "note": "Aucune combinaison ne franchit les filtres de probabilité, valeur et intégrité."}
    total_odds = float(np.prod([s["odds"] for s in chosen]))
    joint_prob = float(np.prod([s["prob"] for s in chosen]))
    return {"selections": chosen, "odds": total_odds, "prob": joint_prob,
            "note": "Probabilité indicative sous hypothèse d'indépendance. Les jambes faibles ou suspectes sont exclues."}


def red_match_profile(score: int, reasons: list) -> dict:
    """Classifies observed anomaly patterns without claiming a real manipulation method."""
    text = " ".join(reasons).lower()
    patterns = []
    if "mouvement" in text or "cote" in text:
        patterns.append("pression de marché / mouvement de cotes")
    if "dispersion" in text:
        patterns.append("désaccord inhabituel entre bookmakers")
    if "modèle/marché" in text or "modèle / marché" in text:
        patterns.append("décalage modèle-marché")
    if "compétition" in text or "amicale" in text or "faible visibilité" in text:
        patterns.append("contexte de compétition à risque")
    if not patterns:
        patterns.append("anomalie indéterminée")
    if score >= 70:
        action = "SURVEILLANCE RENFORCÉE / NO BET"
    elif score >= 50:
        action = "SURVEILLER / NO BET PAR DÉFAUT"
    else:
        action = "SURVEILLANCE"
    return {"patterns": patterns[:4], "action": action}
