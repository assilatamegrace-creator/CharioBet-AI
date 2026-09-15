import math
from typing import Dict, Optional, Tuple, Any

import numpy as np


def _num(v, default=None):
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception:
        return default


def _nested(obj, paths):
    for path in paths:
        cur = obj
        ok = True
        for key in path.split('.'):
            if isinstance(cur, dict): cur = cur.get(key)
            else: ok = False; break
        if ok and cur not in (None, ""): return cur
    return None


def elo_win_probability(elo1: float, elo2: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo1 - elo2) / 400.0))


def ranking_probability(rank1: Optional[float], rank2: Optional[float]) -> Optional[float]:
    if not rank1 or not rank2 or rank1 <= 0 or rank2 <= 0: return None
    # Better rank = larger probability, but ranking is only one component.
    a = math.log1p(rank2) - math.log1p(rank1)
    return 1.0 / (1.0 + math.exp(-1.35 * a))


def _surface_factor(p: Dict, surface: Optional[str]) -> float:
    if not surface: return 0.0
    s = surface.lower()
    stats = p.get("surface_stats") or p.get("surfaceStats") or p.get("stats", {}).get("surface") or {}
    if not isinstance(stats, dict): return 0.0
    block = stats.get(s) or stats.get(s.replace(" ", "_")) or {}
    if not isinstance(block, dict): return 0.0
    win = _num(_nested(block, ["win_pct", "winPercentage", "winRate", "win_rate"]))
    if win is None: return 0.0
    return float(np.clip((win - 50.0) / 100.0, -0.12, 0.12))


def _form_signal(p: Dict) -> float:
    # Accept several common API shapes. Recent match records are optional.
    form = p.get("recent_form") or p.get("recentForm") or p.get("form") or p.get("recent_matches") or []
    if not isinstance(form, list) or not form: return 0.0
    vals = []
    for m in form[:10]:
        if isinstance(m, str): vals.append(1.0 if m.upper().startswith("W") else 0.0)
        elif isinstance(m, dict):
            result = str(m.get("result") or m.get("outcome") or m.get("status") or "").lower()
            if result in {"w", "win", "won", "winner", "victory"}: vals.append(1.0)
            elif result in {"l", "loss", "lost", "loser", "defeat"}: vals.append(0.0)
    if not vals: return 0.0
    return float(np.mean(vals) - 0.5) * 0.24


def _fatigue_signal(p: Dict) -> float:
    # More recent matches / minutes can reduce reliability, not necessarily ability.
    fatigue = _num(_nested(p, ["fatigue", "fatigue_score", "workload"]))
    if fatigue is not None: return float(np.clip(-fatigue / 100.0 * 0.10, -0.10, 0.0))
    recent = p.get("recent_matches") or p.get("recentMatches") or []
    if isinstance(recent, list) and len(recent) >= 5: return -0.025
    return 0.0


def _h2h_signal(p1: Dict, p2: Dict, h2h: Optional[Dict]) -> float:
    if not isinstance(h2h, dict): return 0.0
    p1w = _num(_nested(h2h, ["p1_wins", "player1_wins", "player1Wins", "wins1"]))
    p2w = _num(_nested(h2h, ["p2_wins", "player2_wins", "player2Wins", "wins2"]))
    if p1w is None or p2w is None or p1w + p2w <= 0: return 0.0
    return float(np.clip((p1w / (p1w + p2w) - 0.5) * 0.14, -0.14, 0.14))


def _stat_signal(p: Dict, keys, default=0.0):
    v = _num(_nested(p, keys))
    if v is None: return default
    return v


def tennis_probability(p1: Dict, p2: Dict, surface: Optional[str] = None, h2h: Optional[Dict] = None,
                       tournament: Optional[str] = None) -> Dict[str, float]:
    e1 = _num(_nested(p1, ["stats.ratings.elo", "elo", "ratings.elo"]), 1500.0)
    e2 = _num(_nested(p2, ["stats.ratings.elo", "elo", "ratings.elo"]), 1500.0)
    base = elo_win_probability(e1, e2)
    pr = ranking_probability(_num(p1.get("ranking")), _num(p2.get("ranking")))
    components = {"elo": base, "ranking": pr if pr is not None else base}
    p = 0.68 * base + (0.18 * pr if pr is not None else 0.0)
    weight = 0.86 if pr is not None else 0.68
    # Surface, recent form, H2H and fatigue are deliberately small adjustments.
    adj = _surface_factor(p1, surface) - _surface_factor(p2, surface)
    adj += _form_signal(p1) - _form_signal(p2)
    adj += _fatigue_signal(p1) - _fatigue_signal(p2)
    adj += _h2h_signal(p1, p2, h2h)
    p = p / max(weight, 1e-9) + adj
    p = float(np.clip(p, 0.03, 0.97))
    # Confidence is lower when key data are missing or the matchup is very close.
    completeness = sum([
        pr is not None,
        bool(surface),
        isinstance(h2h, dict) and bool(h2h),
        bool(p1.get("recent_form") or p1.get("recentForm") or p1.get("form")),
        bool(p2.get("recent_form") or p2.get("recentForm") or p2.get("form")),
        _nested(p1, ["stats.service", "service", "serve"]) is not None,
        _nested(p2, ["stats.service", "service", "serve"]) is not None,
    ])
    uncertainty = float(np.clip(0.28 - completeness * 0.025 + (0.08 if abs(p - 0.5) < 0.08 else 0), 0.10, 0.30))
    return {"1": p, "2": 1.0 - p, "elo1": e1, "elo2": e2,
            "confidence": 1.0 - uncertainty, "uncertainty": uncertainty,
            "surface": surface or "inconnue", "data_completeness": completeness}


def _best_of_probs(p1, p2, best_of):
    if best_of == "BO5":
        return {
            "J1 3-0": p1**3,
            "J1 3-1": 3*p1**3*p2,
            "J1 3-2": 6*p1**3*p2**2,
            "J2 3-0": p2**3,
            "J2 3-1": 3*p2**3*p1,
            "J2 3-2": 6*p2**3*p1**2,
        }
    return {"J1 2-0": p1**2, "J1 2-1": 2*p1**2*p2,
            "J2 2-0": p2**2, "J2 2-1": 2*p2**2*p1}


def tennis_markets(prob: Dict[str, float], best_of: str = "BO3") -> Dict[str, float]:
    p1, p2 = prob["1"], prob["2"]
    bo5 = best_of == "BO5"
    n = 5 if bo5 else 3
    # Independent-set approximation. This is a probability engine, not a calibrated bookmaker clone.
    markets = {"Victoire joueur 1": p1, "Victoire joueur 2": p2,
               "J1 gagne au moins un set": 1 - p2**n,
               "J2 gagne au moins un set": 1 - p1**n,
               "Match en 2 sets": (p1**2 + p2**2) if not bo5 else 0.0,
               "Match en 3 sets": (2*p1**2*p2 + 2*p2**2*p1) if not bo5 else 0.0,
               "Match en 4 sets": 0.0 if not bo5 else (3*p1**3*p2 + 3*p2**3*p1),
               "Match en 5 sets": 0.0 if not bo5 else (6*p1**3*p2**2 + 6*p2**3*p1**2),
               "Over 2.5 sets": 1 - (p1**2 + p2**2) if not bo5 else 1.0,
               "Under 2.5 sets": (p1**2 + p2**2) if not bo5 else 0.0,
               "Over 3.5 sets": 0.0 if not bo5 else 1 - (p1**3 + p2**3),
               "Under 3.5 sets": 0.0 if not bo5 else (p1**3 + p2**3),
               "J1 gagne le 1er set": p1,
               "J2 gagne le 1er set": p2}
    markets.update(_best_of_probs(p1, p2, best_of))
    return {k: float(np.clip(v, 0, 1)) for k, v in markets.items() if v > 0}


def tennis_market_candidates(prob: Dict[str, float], odds: Optional[Dict[str, float]] = None,
                             best_of: str = "BO3"):
    rows = []
    for market, p in tennis_markets(prob, best_of).items():
        odd = odds.get(market) if odds else None
        ev = p * odd - 1 if odd and odd > 1 else None
        risk = "faible" if p >= .80 else "modéré" if p >= .68 else "élevé" if p >= .55 else "très élevé"
        if "score" in market.lower() or "3-2" in market or "2-1" in market: risk = "très élevé"
        rows.append({"Marché": market, "Probabilité": p, "Cote": odd, "EV": ev, "Risque": risk})
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Score"] = df["Probabilité"] + df["EV"].fillna(0).clip(-.25,.25)*.30
        df = df.sort_values(["Score","Probabilité"], ascending=False)
    return df.reset_index(drop=True)


def tennis_best_pick(prob: Dict[str, float], best_of: str = "BO3", odds: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
    df = tennis_market_candidates(prob, odds, best_of)
    if df.empty: return "Pas de marché", 0.0
    if odds:
        valid = df[df["Cote"].notna() & (df["Cote"] > 1)]
        if not valid.empty:
            valid = valid.sort_values(["EV","Probabilité"], ascending=False)
            return str(valid.iloc[0]["Marché"]), float(valid.iloc[0]["Probabilité"])
    return str(df.iloc[0]["Marché"]), float(df.iloc[0]["Probabilité"])


def tennis_integrity_score(match: Dict, market_p1: Optional[float] = None,
                           market_odds: Optional[Dict[str,float]] = None,
                           previous_odds: Optional[Dict[str,float]] = None) -> Tuple[int, list]:
    """Suspicion/anomaly score only. It never proves match-fixing."""
    score, reasons = 0, []
    players = match.get("players") or {}
    p1, p2 = players.get("p1") or {}, players.get("p2") or {}
    p = tennis_probability(p1, p2, match.get("surface"))
    if market_p1 is not None:
        diff = abs(p["1"] - market_p1)
        if diff >= .20: score += 35; reasons.append("Écart modèle/marché très important")
        elif diff >= .12: score += 25; reasons.append("Écart modèle/marché notable")
        elif diff >= .08: score += 12; reasons.append("Écart modèle/marché")
    if market_odds and previous_odds:
        changes=[]
        for k,v in market_odds.items():
            old=previous_odds.get(k) if previous_odds else None
            if old and v and old>1 and v>1: changes.append(abs(v/old-1))
        if changes:
            mx=max(changes)
            if mx>=.20: score+=30; reasons.append(f"Mouvement de cote brutal ({mx*100:.0f}%)")
            elif mx>=.12: score+=20; reasons.append(f"Mouvement de cote important ({mx*100:.0f}%)")
            elif mx>=.08: score+=10; reasons.append(f"Mouvement de cote notable ({mx*100:.0f}%)")
    tour=str(match.get("tour") or "").lower(); tournament=str(match.get("tournament") or "").lower()
    if tour in {"itf","juniors"}: score += 10; reasons.append("Circuit à visibilité plus faible")
    if "exhibition" in tournament or "friendly" in tournament: score += 10; reasons.append("Événement exhibition/amical")
    if not p1.get("ranking") or not p2.get("ranking"): score += 5; reasons.append("Classement incomplet")
    if p.get("uncertainty", .3) > .23: score += 8; reasons.append("Incertitude élevée: données incomplètes")
    if not reasons: reasons.append("Aucun signal fort avec les données disponibles")
    return min(100, int(score)), reasons


def tennis_scenario_report(prob: Dict[str,float], best_of: str, p1: Dict, p2: Dict) -> Dict[str, Any]:
    p=prob["1"]
    upset=1-max(p,1-p)
    scenarios=[]
    if abs(p-.5)<.08: scenarios.append("Match très équilibré: variance élevée")
    if prob.get("uncertainty",.2)>.22: scenarios.append("Données incomplètes: réduire la confiance")
    if _fatigue_signal(p1)<-0.02 or _fatigue_signal(p2)<-0.02: scenarios.append("Fatigue potentielle à surveiller")
    scenarios += ["Blessure, abandon, météo/surface et état physique peuvent invalider une projection pré-match"]
    return {"upset_floor": float(upset), "scenarios": scenarios}
