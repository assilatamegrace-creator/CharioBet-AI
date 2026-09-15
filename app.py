import os, re, json, zlib, base64, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from core_tennis import (tennis_probability, tennis_markets, tennis_best_pick, tennis_integrity_score,
                          tennis_market_candidates, tennis_scenario_report)

from core import (fit_goal_model, probabilities, market_candidates, choose_best_market,
                  integrity_assessment, classify_integrity, build_combo, normalize_1x2_odds,
                  devig_market_consensus, market_anchored_prob, decision_grade,
                  select_consensus_market, combo_risk_adjusted, red_match_profile)

st.set_page_config(page_title="CharioBet AI", page_icon="⚽", layout="wide")

DATA_B64 = """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"""

@st.cache_data
def load_data():
    raw = zlib.decompress(base64.b64decode(DATA_B64)).decode("utf-8")
    df = pd.DataFrame(json.loads(raw))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["FTHG", "FTAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)

def pct(x): return f"{float(x)*100:.1f}%"


def init_ledger():
    st.session_state.setdefault("prediction_ledger", [])

def log_prediction(sport, match, market, probability, odds, verdict, score, integrity, source="manuel"):
    init_ledger()
    rec = {
        "Horodatage UTC": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "Sport": sport, "Match": match, "Marché": market,
        "Probabilité modèle": round(float(probability), 4) if probability is not None else None,
        "Cote": round(float(odds), 3) if odds is not None else None,
        "EV": round(float(probability) * float(odds) - 1, 4) if probability is not None and odds and odds > 1 else None,
        "Verdict": verdict, "Score décision": score, "Intégrité": integrity, "Source": source,
        "Résultat": "À régler", "Profit unité": None
    }
    st.session_state["prediction_ledger"].append(rec)

def ledger_dataframe():
    init_ledger()
    return pd.DataFrame(st.session_state["prediction_ledger"])

def render_ledger():
    st.divider()
    st.subheader("📒 Journal de prédictions")
    df = ledger_dataframe()
    if df.empty:
        st.info("Aucune prédiction enregistrée dans cette session.")
        return
    c1,c2,c3 = st.columns(3)
    c1.metric("Prédictions", len(df))
    settled = df[df["Résultat"].isin(["Gagné", "Perdu"])].copy()
    if not settled.empty:
        wins = int((settled["Résultat"] == "Gagné").sum())
        profit = pd.to_numeric(settled["Profit unité"], errors="coerce").fillna(0).sum()
        c2.metric("Taux de réussite", pct(wins/len(settled)))
        c3.metric("Profit (unités)", f"{profit:+.2f}")
    else:
        c2.metric("Taux de réussite", "—")
        c3.metric("Profit (unités)", "—")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Exporter le journal CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="chariobet_journal_predictions.csv", mime="text/csv", use_container_width=True)
    st.caption("Le journal est volontairement séparé des résultats : une prédiction doit être enregistrée avant le résultat pour éviter le cherry-picking.")

def get_api_key():
    try:
        return st.secrets.get("API_FOOTBALL_KEY", os.getenv("API_FOOTBALL_KEY", ""))
    except Exception:
        return os.getenv("API_FOOTBALL_KEY", "")

@st.cache_data(ttl=60)
def api_get_cached(path, params_tuple=()):
    key = get_api_key()
    if not key:
        return None, "Clé API-Football absente"
    try:
        import requests
        params = dict(params_tuple)
        r = requests.get("https://v3.football.api-sports.io/" + path,
                         headers={"x-apisports-key": key}, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            return None, str(payload["errors"])
        return payload.get("response", []), None
    except Exception as e:
        return None, f"Erreur API: {e}"

def api_get(path, params=None):
    return api_get_cached(path, tuple(sorted((params or {}).items())))

def api_fixtures(date_str, live=False):
    return api_get("fixtures", {"live": "all"} if live else {"date": date_str})

def parse_fixture_list(items):
    rows = []
    for x in items or []:
        f, teams, league = x.get("fixture", {}), x.get("teams", {}), x.get("league", {})
        rows.append({
            "fixture_id": f.get("id"), "Date": f.get("date"),
            "status": (f.get("status") or {}).get("short", ""),
            "HomeTeam": (teams.get("home") or {}).get("name", ""),
            "AwayTeam": (teams.get("away") or {}).get("name", ""),
            "League": league.get("name", ""), "Country": league.get("country", ""),
        })
    return pd.DataFrame(rows)

def parse_odds_bookmaker(bookmaker):
    out = {}
    for bet in bookmaker.get("bets", []):
        name = (bet.get("name") or "").lower()
        for v in bet.get("values", []):
            label = str(v.get("value", ""))
            try: odd = float(v.get("odd"))
            except Exception: continue
            if odd <= 1: continue
            if name in ("match winner", "fulltime result") and label in ("Home", "Draw", "Away"):
                out[{"Home":"1", "Draw":"X", "Away":"2"}[label]] = odd
            elif "goals over/under" in name or name == "over/under":
                m = re.match(r"(Over|Under)\\s+(\\d+(?:\\.\\d+)?)", label, re.I)
                if m: out[f"{m.group(1).title()} {float(m.group(2)):g}"] = odd
            elif "both teams to score" in name or "both teams score" in name:
                if label.lower() in ("yes", "no"):
                    out["BTTS Oui" if label.lower()=="yes" else "BTTS Non"] = odd
            elif "double chance" in name:
                if label in ("Home/Draw", "Draw/Home"): out["1X"] = odd
                elif label in ("Draw/Away", "Away/Draw"): out["X2"] = odd
                elif label in ("Home/Away", "Away/Home"): out["12"] = odd
    return out

def parse_odds_pages(items):
    result = {}
    for item in items or []:
        fid = (item.get("fixture") or {}).get("id")
        if not fid: continue
        books = []
        for b in item.get("bookmakers", []):
            parsed = parse_odds_bookmaker(b)
            if parsed:
                books.append({"name": b.get("name", ""), "odds": parsed})
        result[int(fid)] = books
    return result

def api_odds_for_date(date_str):
    all_items = []
    # API-Football paginates odds. Keep a safe cap so the free quota is not murdered by enthusiasm.
    for page in range(1, 4):
        items, err = api_get("odds", {"date": date_str, "page": page})
        if err: return {}, err
        if not items: break
        all_items.extend(items)
        # If fewer than the usual page size arrive, we reached the end.
        if len(items) < 10: break
    return parse_odds_pages(all_items), None

def preferred_odds(bookmakers):
    if not bookmakers: return {}, None
    ordered = sorted(bookmakers, key=lambda x: (0 if "betclic" in x.get("name", "").lower() else 1, x.get("name", "")))
    return ordered[0]["odds"], ordered[0].get("name")

def previous_snapshot(fid):
    return st.session_state.get("odds_snapshots", {}).get(str(fid))

def save_snapshot(fid, odds):
    st.session_state.setdefault("odds_snapshots", {})[str(fid)] = dict(odds)

def integrity_for_fixture(row, bookmakers, data):
    bookmakers = sorted(bookmakers, key=lambda x: (0 if "betclic" in x.get("name", "").lower() else 1, x.get("name", "")))
    odds, book_name = preferred_odds(bookmakers)
    model = fit_goal_model(data, row["HomeTeam"], row["AwayTeam"])
    probs = probabilities(model)
    market_probs = normalize_1x2_odds(odds)
    prev = previous_snapshot(row["fixture_id"])
    quality_home = len(data[(data.HomeTeam == row["HomeTeam"]) | (data.AwayTeam == row["HomeTeam"])])
    quality_away = len(data[(data.HomeTeam == row["AwayTeam"]) | (data.AwayTeam == row["AwayTeam"])])
    quality = min(1.0, (quality_home + quality_away) / 30)
    score, reasons, _ = integrity_assessment(
        probs, market_probs,
        bookmaker_odds=[b["odds"] for b in bookmakers],
        previous_odds=prev,
        league=row.get("League", ""),
        data_quality=quality,
    )
    save_snapshot(row["fixture_id"], odds)
    return {"score": score, "label": classify_integrity(score), "reasons": reasons,
            "odds": odds, "bookmaker": book_name, "model": model, "probs": probs,
            "market_probs": market_probs, "books": len(bookmakers), "history": quality}

data = load_data()
api_key = get_api_key()

tab_football, tab_tennis = st.tabs(["⚽ Football", "🎾 Tennis"])

with tab_tennis:
    st.title("🎾 CharioBet AI Tennis")
    st.caption("Moteur tennis avancé • forme • surface • H2H • classement • Elo • fatigue • marchés • intégrité")
    st.warning("⚠️ L'Integrity Score détecte des anomalies compatibles avec un risque d'intégrité. Il ne peut jamais prouver à lui seul qu'un match est truqué.")

    try:
        tennis_key = st.secrets.get("LIVE_TENNIS_API_KEY", os.getenv("LIVE_TENNIS_API_KEY", ""))
    except Exception:
        tennis_key = os.getenv("LIVE_TENNIS_API_KEY", "")

    @st.cache_data(ttl=120)
    def tennis_api(path, params_tuple=()):
        if not tennis_key: return None, "Clé LIVE_TENNIS_API_KEY absente"
        try:
            import requests
            r=requests.get("https://api.livetennisapi.com/api/public/v1/"+path.lstrip("/"),
                           headers={"X-API-Key":tennis_key}, params=dict(params_tuple), timeout=15)
            payload=r.json()
            if r.status_code>=400: return None, str(payload.get("detail") or payload.get("error") or f"HTTP {r.status_code}")
            return payload.get("data", payload), None
        except Exception as e: return None, f"Erreur API Tennis: {e}"

    if not tennis_key:
        st.info("Ajoute **LIVE_TENNIS_API_KEY** dans Streamlit Secrets pour activer le tennis automatique.")
    else:
        st.success("🟢 API Tennis connectée")
        c1,c2,c3=st.columns(3)
        with c1: tennis_live=st.checkbox("🔴 Live uniquement", key="tennis_live_only")
        with c2: tennis_hours=st.slider("Prochaines heures",2,24,8,key="tennis_hours")
        with c3:
            if st.button("🔄 Actualiser tennis",key="refresh_tennis",use_container_width=True): st.cache_data.clear(); st.rerun()

        status="live" if tennis_live else "upcoming"
        titems,terr=tennis_api("matches",tuple(sorted({"status":status,"limit":100}.items())))
        if terr: st.error(terr); titems=[]
        now=pd.Timestamp.now(tz="UTC"); end=now+pd.Timedelta(hours=tennis_hours)
        tennis_rows=[]
        for m in titems or []:
            if not isinstance(m,dict): continue
            start_t=pd.to_datetime(m.get("scheduled_time"),utc=True,errors="coerce")
            live=str(m.get("status"))=="live"
            if not tennis_live and not (live or (pd.notna(start_t) and now<=start_t<=end)): continue
            players=m.get("players") or {}; p1=players.get("p1") or {}; p2=players.get("p2") or {}
            surface=m.get("surface") or (m.get("tournament_data") or {}).get("surface")
            h2h=m.get("h2h") or m.get("head_to_head")
            prob=tennis_probability(p1,p2,surface,h2h,m.get("tournament"))
            integrity,reasons=tennis_integrity_score(m)
            best_of=str(m.get("format") or m.get("best_of") or "BO3").upper()
            pick,pickprob=tennis_best_pick(prob,best_of)
            tennis_rows.append({"match_id":m.get("id"),"Match":f"{p1.get('name','Joueur 1')} - {p2.get('name','Joueur 2')}",
                "Tournoi":m.get("tournament","-"),"Tour":str(m.get("tour") or "-").upper(),"Surface":surface or "-",
                "Heure UTC":start_t.strftime("%H:%M") if pd.notna(start_t) else ("LIVE" if live else "-"),
                "Statut":"🔴 LIVE" if live else "🕐 À venir","P1":prob["1"],"P2":prob["2"],"Confiance":prob["confidence"],
                "Marché conseillé":pick,"Prob marché":pickprob,"Intégrité":integrity,"Signaux":" • ".join(reasons[:3]),"raw":m})

        if not tennis_rows: st.info("Aucun match tennis dans la fenêtre sélectionnée.")
        else:
            tdf=pd.DataFrame(tennis_rows)
            flagged=tdf[tdf["Intégrité"]>=50].sort_values("Intégrité",ascending=False)
            st.markdown("### 🔴 MATCHS ROUGES TENNIS — anomalies d’intégrité")
            if flagged.empty: st.success("Aucun signal d'anomalie notable avec les données accessibles.")
            else:
                show=flagged.copy(); show["P1"]=show["P1"].map(pct); show["P2"]=show["P2"].map(pct); show["Confiance"]=show["Confiance"].map(pct)
                st.dataframe(show[["Match","Tournoi","Tour","Surface","Heure UTC","P1","P2","Confiance","Intégrité","Signaux"]],hide_index=True,use_container_width=True)
                st.caption("🔴 MATCH ROUGE = anomalie/suspicion, pas preuve de trucage. CharioBet affiche une projection sportive indépendante, mais bloque la recommandation automatique et le combiné.")
                for _, tr in flagged.head(10).iterrows():
                    try:
                        raw=tr["raw"]; players=raw.get("players") or {}; pp1=players.get("p1") or {}; pp2=players.get("p2") or {}
                        bo=str(raw.get("format") or raw.get("best_of") or "BO3").upper()
                        td=tennis_probability(pp1,pp2,raw.get("surface"),raw.get("h2h") or raw.get("head_to_head"),raw.get("tournament"))
                        tm=tennis_markets(td,bo)
                        ranked=sorted([(v,k) for k,v in tm.items() if v>=0.55],reverse=True)[:4]
                        st.markdown(f"**🔴 {tr['Match']}** · Integrity {int(tr['Intégrité'])}/100")
                        st.write("Projection anti-piège, hors mouvements de cotes : " + " · ".join(f"{k}: {pct(v)}" for v,k in ranked))
                        st.write("Décision : **NO BET / surveillance renforcée**. Une manipulation inconnue peut rendre une projection sportive non fiable.")
                    except Exception:
                        pass

            st.markdown("### 🧠 Analyse automatique haut niveau")
            picks=[]
            for r in tennis_rows:
                raw=r["raw"]; players=raw.get("players") or {}; p1=players.get("p1") or {}; p2=players.get("p2") or {}
                prob=tennis_probability(p1,p2,r["Surface"] if r["Surface"]!="-" else None,raw.get("h2h") or raw.get("head_to_head"),raw.get("tournament"))
                bo=str(raw.get("format") or raw.get("best_of") or "BO3").upper()
                pick,pp=tennis_best_pick(prob,bo)
                scenario=tennis_scenario_report(prob,bo,p1,p2)
                picks.append({"Match":r["Match"],"Tournoi":r["Tournoi"],"Surface":r["Surface"],"Marché":pick,
                              "Probabilité":pp,"Confiance":prob["confidence"],"Risque upset":scenario["upset_floor"],"Intégrité":r["Intégrité"]})
            pdf=pd.DataFrame(picks).sort_values(["Confiance","Probabilité"],ascending=False)
            for c in ["Probabilité","Confiance","Risque upset"]: pdf[c]=pdf[c].map(pct)
            st.dataframe(pdf,hide_index=True,use_container_width=True)
            st.caption("Le moteur pénalise la confiance lorsque surface, forme, H2H ou statistiques sont absents. Il ne promet jamais 80% de gains: une probabilité modèle n'est pas une garantie.")

    st.divider(); st.markdown("### 🧪 Analyse manuelle complète d'un match tennis")
    if tennis_key:
        pcol1,pcol2=st.columns(2)
        with pcol1: player1_name=st.text_input("Joueur 1",placeholder="Ex. Carlos Alcaraz",key="tennis_p1")
        with pcol2: player2_name=st.text_input("Joueur 2",placeholder="Ex. Jannik Sinner",key="tennis_p2")
        if st.button("🔎 ANALYSER LE MATCH TENNIS",type="primary",use_container_width=True):
            if len(player1_name.strip())<3 or len(player2_name.strip())<3: st.error("Entre les deux noms de joueurs.")
            else:
                a1,e1=tennis_api("players",tuple(sorted({"search":player1_name.strip(),"limit":5}.items())))
                a2,e2=tennis_api("players",tuple(sorted({"search":player2_name.strip(),"limit":5}.items())))
                if e1 or e2 or not a1 or not a2: st.error(e1 or e2 or "Joueur introuvable.")
                else:
                    pp1,pp2=a1[0],a2[0]
                    # Try to enrich the player matchup with H2H. If unavailable, continue honestly.
                    h2h=None
                    id1=pp1.get("id") or pp1.get("player_id"); id2=pp2.get("id") or pp2.get("player_id")
                    if id1 and id2:
                        h2h_data,herr=tennis_api(f"h2h/{id1}/{id2}")
                        if not herr: h2h=h2h_data
                    surface=st.selectbox("Surface",["hard","clay","grass","indoor hard","inconnue"],key="manual_surface")
                    bo=st.selectbox("Format",["BO3","BO5"],key="manual_bo")
                    d=tennis_probability(pp1,pp2,None if surface=="inconnue" else surface,h2h)
                    st.markdown(f"### {pp1.get('name')} vs {pp2.get('name')}")
                    x,y,z=st.columns(3); x.metric(pp1.get("name","Joueur 1"),pct(d["1"])); y.metric(pp2.get("name","Joueur 2"),pct(d["2"])); z.metric("Confiance",pct(d["confidence"]))
                    x,y,z=st.columns(3); x.metric("Elo J1",f"{d['elo1']:.0f}"); y.metric("Elo J2",f"{d['elo2']:.0f}"); z.metric("Incertitude",pct(d["uncertainty"]))
                    st.markdown("#### 🎯 Marchés analysés")
                    md=tennis_market_candidates(d,None,bo); md["Probabilité"]=md["Probabilité"].map(pct)
                    st.dataframe(md[["Marché","Probabilité","Risque"]],hide_index=True,use_container_width=True)
                    scenario=tennis_scenario_report(d,bo,pp1,pp2)
                    st.markdown("#### 🌪️ Scénarios imprévus")
                    for s in scenario["scenarios"]: st.write("• "+s)
                    st.markdown("#### 🛡️ Intégrité")
                    integ,reasons=tennis_integrity_score({"players":{"p1":pp1,"p2":pp2},"surface":surface})
                    label="🔴 surveillance élevée" if integ>=70 else "🟠 anomalie notable" if integ>=50 else "🟡 à surveiller" if integ>=30 else "🟢 aucun signal fort"
                    st.write(f"**{integ}/100 • {label}**")
                    st.write(" • ".join(reasons))
                    st.info("Le système cherche à réduire les erreurs en combinant plusieurs signaux et en abaissant la confiance quand des données essentielles manquent. Il ne peut pas garantir 80% de victoires ni éliminer les imprévus.")

with tab_football:
    api_key = get_api_key()

    st.caption("V0.9 • décision stricte • journal de prédictions • Match Rouge • combinés multi-matchs")
    st.warning("⚠️ Le module d'intégrité repère des anomalies de marché et de données. Il ne peut pas prouver qu'un match est truqué. Une alerte sérieuse doit être vérifiée par des données professionnelles et, idéalement, par un organisme d'intégrité.")

    with st.sidebar:
        st.header("⚙️ État du moteur")
        if api_key: st.success("🟢 API-Football connectée")
        else: st.error("🔴 API-Football absente")
        st.caption(f"Historique embarqué : {len(data)} matchs")
        st.caption("Volume réel des mises : non disponible avec cette API")

    # ---------------- AUTO SCAN ----------------
    st.subheader("🚨 Surveillance automatique des matchs")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Actualiser le scan", use_container_width=True):
            st.cache_data.clear(); st.rerun()
    with col2:
        live_only = st.checkbox("🔴 Live uniquement")
    with col3:
        hours = st.slider("Prochaines heures", 2, 24, 8)

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    fixtures = pd.DataFrame()
    if api_key:
        items, err = api_fixtures(today, live=live_only)
        fixtures = parse_fixture_list(items)
        if err: st.error(err)
        if not fixtures.empty:
            fixtures["DateUTC"] = pd.to_datetime(fixtures["Date"], utc=True, errors="coerce")
            now = pd.Timestamp.now(tz="UTC")
            live_status = fixtures["status"].isin(["1H","HT","2H","ET","BT","P","LIVE"])
            if not live_only:
                end = now + pd.Timedelta(hours=hours)
                fixtures = fixtures[(live_status) | ((fixtures["DateUTC"] >= now) & (fixtures["DateUTC"] <= end))]
            else:
                fixtures = fixtures[live_status]
            fixtures = fixtures.sort_values("DateUTC").reset_index(drop=True)
    else:
        st.info("Ajoute API_FOOTBALL_KEY dans Streamlit Secrets pour activer le scan automatique.")

    if fixtures.empty:
        st.info("Aucun match dans la fenêtre sélectionnée.")
    else:
        odds_map, odds_err = api_odds_for_date(today) if api_key else ({}, None)
        if odds_err: st.warning(odds_err)
        scan_rows = []
        for _, row in fixtures.iterrows():
            try:
                assessment = integrity_for_fixture(row, odds_map.get(int(row["fixture_id"]), []), data)
                scan_rows.append({
                    "Match": f"{row['HomeTeam']} - {row['AwayTeam']}",
                    "Compétition": row["League"],
                    "Heure UTC": pd.to_datetime(row["Date"], utc=True).strftime("%H:%M"),
                    "Intégrité": assessment["score"],
                    "Statut": assessment["label"],
                    "Bookmakers": assessment["books"],
                    "Cote source": assessment["bookmaker"] or "-",
                    "Signaux": " • ".join(assessment["reasons"][:2]),
                    "Profil rouge": " • ".join(red_match_profile(assessment["score"], assessment["reasons"])["patterns"]),
                    "fixture_id": row["fixture_id"],
                })
            except Exception as e:
                scan_rows.append({"Match": f"{row['HomeTeam']} - {row['AwayTeam']}", "Compétition": row["League"],
                                  "Heure UTC": "-", "Intégrité": 0, "Statut": "⚪ Données insuffisantes",
                                  "Bookmakers": 0, "Cote source": "-", "Signaux": str(e), "fixture_id": row["fixture_id"]})

        scan_df = pd.DataFrame(scan_rows)
        flagged = scan_df[scan_df["Intégrité"] >= 50].sort_values("Intégrité", ascending=False)
        normal = scan_df[scan_df["Intégrité"] < 50].sort_values("Intégrité", ascending=False)

        st.markdown("### 🔴 Matchs à surveiller en priorité")
        if flagged.empty:
            st.success("Aucun match ne dépasse actuellement le seuil d'anomalie notable.")
        else:
            st.error(f"{len(flagged)} match(s) classé(s) MATCH ROUGE : signaux d’intégrité à examiner.")
            st.dataframe(flagged.drop(columns=["fixture_id"]), hide_index=True, use_container_width=True)
            st.caption("🔴 MATCH ROUGE = anomalie/suspicion à examiner, jamais preuve de trucage. La projection sportive est séparée du signal d’intégrité.")

        st.markdown("### 🟢 Autres matchs de la fenêtre")
        st.dataframe(normal.drop(columns=["fixture_id"]), hide_index=True, use_container_width=True)

        st.markdown("### 🧪 Ce que CharioBet mesure")
        a,b,c,d = st.columns(4)
        a.metric("Écart modèle / marché", "Oui")
        b.metric("Dispersion bookmakers", "Oui")
        c.metric("Mouvement de cote", "Session")
        d.metric("Volume des mises", "Non disponible")
        st.caption("Le mouvement est comparé aux snapshots vus pendant cette session. L'API-Football conserve les cotes pré-match sur une fenêtre limitée et ne fournit pas l'intelligence client/volume nécessaire à une vraie plateforme d'intégrité.")

    # ---------------- MANUAL ----------------
    st.divider()
    st.subheader("🎯 Analyse détaillée d’un match")
    teams = sorted(set(data.HomeTeam) | set(data.AwayTeam))
    c1,c2 = st.columns(2)
    with c1: home = st.selectbox("Domicile", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    with c2:
        opts=[t for t in teams if t != home]
        away=st.selectbox("Extérieur", opts, index=opts.index("Chelsea") if "Chelsea" in opts else 0)
    o1,o2,o3=st.columns(3)
    with o1: odd1=st.number_input("Cote 1", min_value=0.0, value=2.0, step=.01)
    with o2: oddx=st.number_input("Cote X", min_value=0.0, value=3.4, step=.01)
    with o3: odd2=st.number_input("Cote 2", min_value=0.0, value=3.5, step=.01)

    if st.button("🔎 ANALYSER", type="primary", use_container_width=True):
        try:
            model=fit_goal_model(data,home,away); probs=probabilities(model)
            market_probs={}
            if odd1>1 and oddx>1 and odd2>1:
                market_probs=normalize_1x2_odds({"1":odd1,"X":oddx,"2":odd2})
            score,reasons,_=integrity_assessment(probs,market_probs,league="",data_quality=1.0)
            st.markdown(f"### {home} vs {away}")
            a,b,c=st.columns(3); a.metric(home,pct(probs["1"])); b.metric("Nul",pct(probs["X"])); c.metric(away,pct(probs["2"]))
            x,y=st.columns(2); x.metric("Buts attendus domicile",f"{model.lambda_home:.2f}"); y.metric("Buts attendus extérieur",f"{model.lambda_away:.2f}")
            md=market_candidates(model,{"1":odd1,"X":oddx,"2":odd2})
            md["Probabilité"]=md["Probabilité"].map(pct); md["EV"]=md["EV"].map(lambda x:"-" if pd.isna(x) else pct(x))
            st.dataframe(md[["Marché","Probabilité","Cote","EV","Risque"]].head(30),hide_index=True,use_container_width=True)
            st.markdown("### 🛡️ Integrity Score")
            if score>=70: st.error(f"{score}/100 • 🔴 Surveillance élevée")
            elif score>=50: st.warning(f"{score}/100 • 🟠 Anomalie notable")
            elif score>=30: st.warning(f"{score}/100 • 🟡 À surveiller")
            else: st.success(f"{score}/100 • 🟢 Aucun signal fort")
            st.write(" • ".join(reasons))
            # Enregistrement uniquement du signal réellement affiché, pour permettre un suivi pré-match.
            best_row = md.iloc[0] if not md.empty else None
            if best_row is not None and pd.notna(best_row.get("Cote")) and float(best_row.get("Cote", 0)) > 1:
                st.session_state["pending_log"] = {
                    "sport": "Football", "match": f"{home} - {away}",
                    "market": str(best_row["Marché"]), "prob": float(md.iloc[0]["Probabilité"].rstrip("%"))/100 if isinstance(md.iloc[0]["Probabilité"], str) else float(best_row["Probabilité"]),
                    "odds": float(best_row["Cote"]), "integrity": int(score),
                }
                if st.button("📌 Enregistrer cette prédiction", key="log_manual_football"):
                    pr = st.session_state.pop("pending_log")
                    ev = pr["prob"]*pr["odds"]-1
                    verdict = "NO BET" if score >= 50 or ev < 0.05 else "SIGNAL VALIDE"
                    log_prediction("Football", pr["match"], pr["market"], pr["prob"], pr["odds"], verdict, None, pr["integrity"])
                    st.success("Prédiction enregistrée avant le résultat.")
        except Exception as e: st.error(f"Analyse impossible : {e}")

    # ---------------- COMBO ----------------
    st.divider()
    st.subheader("🎰 Générateur de combiné multi-matchs")
    profile=st.radio("Choisis le profil",["Prudent","Équilibré","Grosse cote"],horizontal=True)
    st.caption("Chaque sélection vient d’un match différent. CharioBet exclut automatiquement les matchs avec anomalie notable du combiné.")

    if fixtures.empty:
        st.info("Le combiné automatique nécessite des matchs live/à venir avec des cotes.")
    else:
        selections=[]
        for _,row in fixtures.iterrows():
            try:
                books=odds_map.get(int(row["fixture_id"]),[])
                odds,_=preferred_odds(books)
                model=fit_goal_model(data,row["HomeTeam"],row["AwayTeam"])
                best=choose_best_market(model,odds)
                assess=integrity_for_fixture(row,books,data)
                if best and assess["score"]<50:
                    selections.append({"match":f"{row['HomeTeam']} - {row['AwayTeam']}","market":best["Marché"],
                                       "prob":float(best["Probabilité"]),"odds":float(best["Cote"]),
                                       "ev":float(best["EV"] or 0),"integrity":assess["score"]})
            except Exception:
                pass
        combo=combo_risk_adjusted(selections,profile)
        if combo["selections"]:
            st.success(f"Combiné {profile} • {len(combo['selections'])} matchs • cote totale ≈ {combo['odds']:.2f} • probabilité indépendante ≈ {pct(combo['prob'])}")
            st.dataframe(pd.DataFrame(combo["selections"]),hide_index=True,use_container_width=True)
            st.caption(combo["note"])
        else:
            st.warning("Aucun combiné suffisamment solide et sans anomalie notable n’a été trouvé.")

    st.divider()
    st.subheader("🧠 Pourquoi CharioBet ne force plus un pronostic")
    st.write("Le moteur combine modèle statistique, consensus de marché, valeur, qualité des données et intégrité. Si le signal ne passe pas les seuils, il affiche NO BET au lieu d'inventer une certitude.")
    b1,b2,b3,b4=st.columns(4)
    b1.metric("Modèle + marché", "Ensemble")
    b2.metric("Valeur minimale", "+5%")
    b3.metric("Intégrité élevée", "Exclue")
    b4.metric("Prono forcé", "Non")

    st.caption("🔴 Règle MATCH ROUGE : les anomalies sont classées à part, leurs projections sportives sont affichées séparément pour comprendre le scénario, mais elles sont bloquées des recommandations automatiques et des combinés. Les volumes financiers privés et alertes propriétaires ne sont pas accessibles via API-Football.")

render_ledger()
