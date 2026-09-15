# CharioBet AI v0.5

Assistant de pronostics football + tennis avec surveillance d'anomalies.

## Sports
- Football : API-Football, analyses automatiques, marchés, combinés multi-matchs et Integrity Score.
- Tennis : Live Tennis API, matchs live/à venir, joueurs, classement/Elo et probabilités de victoire.

## Secrets Streamlit
```toml
API_FOOTBALL_KEY = "..."
LIVE_TENNIS_API_KEY = "..."
```

La clé tennis est séparée de la clé football. Le niveau gratuit de Live Tennis API fournit les matchs live/à venir, scores, joueurs et fixtures. Les marchés/cotes et certaines données historiques comme le H2H dépendent du niveau d'accès de l'API.

## Modèle tennis v0.5
- Elo du joueur + classement officiel.
- Probabilité victoire joueur 1 / joueur 2.
- Baseline de score en sets BO3/BO5.
- Filtre d'anomalies séparé.

Ce modèle tennis est un **prototype**. Il ne doit pas être présenté comme une garantie de résultat ni comme un détecteur de match truqué.

## Benchmark v0.7
CharioBet borrows useful ideas observed across major prediction/statistics platforms while rejecting their common weaknesses:
- Forebet: broad mathematical coverage -> keep broad coverage, but add calibration and explicit uncertainty.
- FootyStats: many markets and historical/statistical views -> keep market breadth, but rank by measurable edge and refuse weak signals.
- Understat/WhoScored-style analytics: deep performance metrics -> use xG/team/player context when the data source provides them.
- BetRedge/BetValue-style value engines: model probability vs devigged market probability, edge, and transparent reasoning -> add market anchoring and NO BET gates.
- MatchSignal-style backtesting/track records: every future prediction should be logged pre-event and evaluated later; the current deployment must not claim a proven record until enough out-of-sample results exist.
- Tennis Abstract/TennisExplorer-style tennis depth: player ratings, surface, serve/return, H2H and history -> feed these into the tennis engine where available.

Important: competitor claims of accuracy are not treated as truth. CharioBet should publish its own calibration, Brier/log-loss, ROI, drawdown and market-by-market record from timestamped pre-match predictions.
