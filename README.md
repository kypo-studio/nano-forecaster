# nano-forecaster

## Objectif et hypothèse

Ce banc d'essai mesure si une petite attention causale apporte un gain face à des
méthodes classiques pour prévoir `OT` dans ETTh1. L'objectif est méthodologique,
pas une revendication d'état de l'art.

## Données et protocole

ETTh1 contient 17420 observations horaires. Le split temporel est
strict, avec frontières aux indices 12194 et
13935. Le normaliseur est ajusté sur le train uniquement.
Le run rapporté utilise 4096 fenêtres train,
64 validation et 64 test,
avec la seed 42 et la configuration `configs/default.yaml`.

## Architecture

```text
features -> projection -> position -> blocs causaux -> dernier token -> H sorties
                                  | attention multi-têtes + FFN |
```

L'attention, le masque causal, les résiduels, les normalisations et la tête directe
sont codés dans `src/model.py`, sans `nn.Transformer` ni modèle préfabriqué.
Le Transformer compte 159840 paramètres, le MLP
en compte 100640.

## Baselines

Le protocole compare dernier point, naïve saisonnière 24 h, ARIMA, SARIMA, MLP et
XGBoost sur les mêmes origines, cibles et horizons. XGBoost reçoit les retards
multivariés ainsi que l'heure, le jour de semaine et le mois en encodage cyclique.

## Résultats

| model | horizon | MAE | RMSE | MAPE | sMAPE |
| --- | --- | --- | --- | --- | --- |
| ARIMA | 24 | 1.0489 | 1.3334 | 56.8008 | 49.2920 |
| SARIMA | 24 | 1.0629 | 1.3495 | 57.7695 | 50.3507 |
| Naive | 24 | 1.2240 | 1.5502 | 65.3857 | 59.4457 |
| SeasonalNaive | 24 | 1.5282 | 1.8549 | 87.8313 | 68.1129 |
| Transformer | 24 | 4.1853 | 4.3594 | 223.1382 | 95.9831 |
| XGBoost | 24 | 4.3442 | 4.7082 | 231.6889 | 95.2568 |
| MLP | 24 | 5.9748 | 6.2079 | 310.9857 | 111.3227 |
| ARIMA | 48 | 1.1750 | 1.4351 | 58.6731 | 50.5312 |
| SARIMA | 48 | 1.2377 | 1.5115 | 61.8776 | 53.0211 |
| Naive | 48 | 1.5311 | 1.8757 | 75.4211 | 66.7472 |
| SeasonalNaive | 48 | 1.6332 | 1.9098 | 85.1341 | 70.0762 |
| Transformer | 48 | 4.2356 | 4.4268 | 215.2431 | 93.5866 |
| XGBoost | 48 | 5.8135 | 6.2141 | 282.5437 | 105.7228 |
| MLP | 48 | 6.1071 | 6.3852 | 304.8798 | 109.2390 |
| ARIMA | 96 | 1.0949 | 1.3284 | 45.4974 | 42.1078 |
| SARIMA | 96 | 1.1219 | 1.3774 | 47.0109 | 43.2373 |
| SeasonalNaive | 96 | 1.3717 | 1.6954 | 61.4562 | 57.2663 |
| Naive | 96 | 1.4243 | 1.7864 | 59.9222 | 59.2250 |
| Transformer | 96 | 3.9160 | 4.1100 | 176.5223 | 83.5484 |
| MLP | 96 | 5.9132 | 6.1586 | 257.4253 | 101.8592 |
| XGBoost | 96 | 6.9705 | 7.2710 | 289.4379 | 109.0756 |

![Erreur par horizon](results/error_by_horizon.png)

![Réel et prédit](results/forecast.png)

![Loss](results/loss.png)

## Analyse honnête

Meilleur MAE observé par horizon: H=24: ARIMA (MAE 1.0489); H=48: ARIMA (MAE 1.1750); H=96: ARIMA (MAE 1.0949).
Le Transformer ne gagne sur aucun horizon dans ce run. Détail: H=24: écart de 3.1364 MAE face au meilleur modèle; H=48: écart de 3.0606 MAE face au meilleur modèle; H=96: écart de 2.8211 MAE face au meilleur modèle.
Le temps mesuré d'entraînement du Transformer est
10.38 s sur `cpu`.
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
