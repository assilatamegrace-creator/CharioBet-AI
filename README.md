# CharioBet AI v0.3

Prototype Streamlit plus robuste avec :
- modèle Poisson + Dixon-Coles + Elo léger ;
- marchés 1X2, double chance, DNB, over/under, BTTS, équipe marque, clean sheet, win-to-nil, premier but, scores exacts ;
- Integrity Score = proxy d'anomalies, jamais une preuve de match truqué ;
- affichage automatique des matchs du jour / live via API-Football si `API_FOOTBALL_KEY` est configurée ;
- générateur de combinés multi-matchs : prudent, équilibré, grosse cote ;
- un marché peut être différent pour chaque match ;
- historique intégré : 2 660 matchs de Premier League 2019/20–2025/26.

## Déploiement
1. Mettre `app.py`, `core.py`, `requirements.txt`, `README.md` dans le dépôt.
2. Dans Streamlit Cloud, configurer le secret `API_FOOTBALL_KEY` pour activer les données live et les cotes.
3. Entrypoint : `app.py`.

Sans clé API, l'analyse historique/manuelle reste disponible. Les fonctions automatiques live et combiné automatique restent désactivées.
