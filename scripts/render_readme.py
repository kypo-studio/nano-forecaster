#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads((ROOT / "results/metadata.json").read_text())
metrics = (ROOT / "results/metrics.md").read_text().strip()
best = {}
import pandas as pd
table = pd.read_csv(ROOT / "results/metrics.csv")
for horizon, rows in table.groupby("horizon"):
    winner = rows.loc[rows.MAE.idxmin()]
    best[int(horizon)] = (winner["model"], winner["MAE"])
analysis = "; ".join(f"H={h}: {m} (MAE {v:.4f})" for h, (m, v) in best.items())
transformer_comparison = []
for horizon, rows in table.groupby("horizon"):
    transformer_mae = float(rows.loc[rows.model == "Transformer", "MAE"].iloc[0])
    best_mae = float(rows.MAE.min())
    transformer_comparison.append(
        f"H={int(horizon)}: écart de {transformer_mae - best_mae:.4f} MAE face au meilleur modèle"
    )
comparison = "; ".join(transformer_comparison)

readme = f"""# nano-forecaster

## Objectif et hypothèse

Ce banc d'essai mesure si une petite attention causale apporte un gain face à des
méthodes classiques pour prévoir `OT` dans ETTh1. L'objectif est méthodologique,
pas une revendication d'état de l'art.

## Données et protocole

ETTh1 contient {metadata['rows']} observations horaires. Le split temporel est
strict, avec frontières aux indices {metadata['splits']['train_end']} et
{metadata['splits']['val_end']}. Le normaliseur est ajusté sur le train uniquement.
Le run rapporté utilise {metadata['windows']['train']} fenêtres train,
{metadata['windows']['val']} validation et {metadata['windows']['test']} test,
avec la seed {metadata['seed']} et la configuration `{metadata['config']}`.

## Architecture

```text
features -> projection -> position -> blocs causaux -> dernier token -> H sorties
                                  | attention multi-têtes + FFN |
```

L'attention, le masque causal, les résiduels, les normalisations et la tête directe
sont codés dans `src/model.py`, sans `nn.Transformer` ni modèle préfabriqué.
Le Transformer compte {metadata['parameters']['Transformer']} paramètres, le MLP
en compte {metadata['parameters']['MLP']}.

## Baselines

Le protocole compare dernier point, naïve saisonnière 24 h, ARIMA, SARIMA, MLP et
XGBoost sur les mêmes origines, cibles et horizons. XGBoost reçoit les retards
multivariés ainsi que l'heure, le jour de semaine et le mois en encodage cyclique.

## Résultats

{metrics}

![Erreur par horizon](results/error_by_horizon.png)

![Réel et prédit](results/forecast.png)

![Loss](results/loss.png)

## Analyse honnête

Meilleur MAE observé par horizon: {analysis}.
Le Transformer ne gagne sur aucun horizon dans ce run. Détail: {comparison}.
Le temps mesuré d'entraînement du Transformer est
{metadata['training_seconds']['Transformer']:.2f} s sur `{metadata['device']}`.
Ces chiffres décrivent ce run précis et ne sont pas extrapolés. MAPE exclut les
cibles exactement nulles, pour lesquelles cette métrique est indéfinie.

## Limites

Le run publié emploie la configuration rapide et un sous-ensemble déterministe
des fenêtres. ARIMA et SARIMA sont réajustés à chaque origine, ce qui est loyal
mais coûteux. Une seule seed et un seul dataset ne permettent pas d'estimer la
variance ni la généralisation inter-datasets. MAPE est instable près de zéro.

## Reproduire

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh configs/small.yaml
```

Le script télécharge ETTh1, lance les tests, entraîne tous les modèles, recalcule
les métriques, régénère les figures et réécrit ce README depuis les artefacts.
Pour l'expérience plus longue: `bash scripts/run_all.sh configs/default.yaml`.

## Licence

Apache License 2.0, voir `LICENSE`.
"""
if "—" in readme or "–" in readme:
    raise SystemExit("Forbidden dash character in README")
(ROOT / "README.md").write_text(readme)
