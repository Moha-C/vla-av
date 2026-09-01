# Protocole SimLingo + Dreamer/RSSM aligné sur le rapport

Ce protocole décrit ce qui est réellement implémenté et ce qui doit encore être
mesuré. Il ne modifie aucun fichier du rapport.

## Invariants

- SimLingo calcule toujours ses waypoints, ses vitesses et sa commande PID.
- Le candidat 0 est exactement cette commande native post-PID.
- Le RSSM reçoit une observation structurée normalisée de 32 dimensions.
- Sa transition reçoit uniquement la commande physique finale
  `[steer, throttle, brake]`. `alpha` est journalisé séparément et intervient
  dans l'autorité et la récompense, pas comme quatrième entrée dynamique.
- Les autres candidats sont des variations bornées autour de SimLingo.
- Leur futur est déroulé dans l'espace latent, sans exécution préalable dans
  CARLA.
- L'actor et le critic sont entraînés sur des retours imaginés par le RSSM.
- La commande finale est `(1-alpha) * SimLingo + alpha * Dreamer`.
- Aucun veto géométrique caché n'est présent dans C, D ou E. Les anciens guards
  restent isolés dans B.
- Un échec de chargement d'un mode RSSM demandé arrête le run. Il n'existe plus
  de repli silencieux vers la baseline.

## Conditions expérimentales

| Condition | Exécution |
| --- | --- |
| A | SimLingo natif, aucun module Dreamer construit |
| B | ancien Dreamer avec guards, préservé séparément |
| C | RSSM et candidats, alpha fixe faible |
| D | RSSM, actor/critic imaginés et alpha continu appris |
| E | D plus calibrateur pairwise séparé |

Le mode shadow exécute le calcul de D mais force `alpha=0`. Il sert au test
d'intégration et n'est jamais compté comme un résultat fermé D.

## 1. Audit des données

La Phase 1 doit d'abord produire des trajectoires strictement natives. Le
collecteur est passif : il observe la commande SimLingo après son PID, mais ne
la modifie jamais. Il peut être lancé depuis la webapp avec le mode
`Report Phase 1 - native SimLingo collect`, Dreamer réglé sur `off`, ou en
ligne de commande :

```bash
cd ~/Desktop/vla-av
ROUTE_ID=57 SEED=20260818 CARLA_QUALITY=Low \
  bash scripts/run_report_dreamer_native_collect.sh
```

Chaque exécution est écrite sous
`data/report_dreamer/native/runs/`. Une trajectoire ne devient admissible
qu'après sa finalisation avec un résultat Bench2Drive frais, `Finished`,
éligible et dont chaque route est `Completed`. Un résultat `Started`,
interrompu, incomplet ou issu d'un crash est refusé au lieu d'être assimilé à
zéro collision. Par défaut,
l'entraînement refuse les traces issues d'un guard, d'une policy RL, d'une
source inconnue ou dépourvues de vérité terrain Bench2Drive. Les options
`--source-policy any` et `--allow-missing-event-ground-truth` sont réservées
aux diagnostics et ne doivent jamais servir à produire le checkpoint de
production.

```bash
cd ~/Desktop/vla-av
~/miniconda3/envs/simlingo/bin/python scripts/train_report_dreamer.py inspect \
  --output checkpoints/report_aligned_dreamer/audit
```

Le manifeste contient chaque trace acceptée/refusée, sa map, sa route, son
scénario, sa seed, son nombre de transitions, la couverture de vérité terrain
et les distributions observées de steer/throttle/brake, alpha et récompense.
Le split est effectué par seed avant la création des fenêtres temporelles. Les
ensembles train, validation et test doivent être disjoints et contenir chacun
au moins deux seeds distinctes. Le vecteur 32D écrit pendant la collecte native
est relu tel quel lorsqu'il est complet ; il n'est pas reconstruit avec des
valeurs par défaut.

## 2. Entraînement offline du RSSM et de l'actor/critic

```bash
cd ~/Desktop/vla-av
~/miniconda3/envs/simlingo/bin/python scripts/train_report_dreamer.py all \
  --config configs/dreamer_report_aligned.yaml \
  --device cuda \
  --output checkpoints/report_aligned_dreamer/candidate
```

Cette commande entraîne d'abord le world model sur les séquences train,
sélectionne son checkpoint sur validation, fige le RSSM, puis entraîne actor et
critic par reinforcement learning dans les trajectoires imaginées. Le fichier
final reste nommé `report_dreamer_candidate.pt` et porte le statut
`candidate_not_promoted`.

### Correctif d'identifiabilité des actions

La variante `configs/dreamer_report_aligned_action_sensitive.yaml` corrige un
défaut observé sur le premier checkpoint : le latent variait avec l'action,
mais les têtes de risque et de collision classaient presque pareil le freinage
fort et l'accélération. Le correctif comprend quatre éléments mesurables :

- la cible de risque de la transition `t -> t+1` est calculée à partir de
  l'état `t+1`, et non de l'état avant l'action ;
- les têtes de prédiction supervisent aussi directement le prior conditionné
  par l'action, en plus du posterior utilisé pour la reconstruction ;
- une perte contrastive demande à l'action réellement observée de rapprocher
  davantage le prior du prochain état latent que des actions alternatives ;
- sur les états manifestement dangereux, un prior physique d'entraînement
  demande au freinage fort de prédire moins de vitesse, de progression et de
  risque que l'accélération forte.

Ce dernier point est une hypothèse inductive déclarée destinée à combattre le
biais observationnel des traces (les freinages sont surtout observés lorsque
le danger est déjà élevé). Ce n'est **pas** un guard en boucle fermée : aucun
candidat n'est imposé, rejeté ou remplacé à l'exécution. Le freinage d'urgence
devient en revanche un candidat explicite que le RSSM doit classer comme les
autres. L'actor neuf démarre avec une autorité de 5 %, afin de rester un
complément de SimLingo avant validation fermée.

## 3. Évaluation figée des têtes RSSM

```bash
~/miniconda3/envs/simlingo/bin/python scripts/evaluate_report_dreamer.py \
  --checkpoint checkpoints/report_aligned_dreamer/candidate/report_dreamer_candidate.pt \
  --manifest checkpoints/report_aligned_dreamer/candidate/dataset_manifest.json \
  --device cuda
```

Le script refuse un autre dataset, un autre split ou une architecture
incompatible. Il mesure séparément reconstruction, progression, risque,
continuation, value, collision et off-road, avec moyenne et dispersion entre
seeds. Il sonde également cinq commandes sur les mêmes états figés : native,
freinage fort, accélération forte, braquage gauche et braquage droit. Le rapport
mesure la dispersion des transitions et des sorties, la fraction de prédictions
effondrées, ainsi que l'avantage de risque/collision du freinage sur
l'accélération dans les états dangereux. Ces erreurs et tests de sensibilité ne
constituent pas une preuve d'amélioration en boucle fermée.

## 4. Calibrateur pairwise optionnel

E nécessite un JSONL réellement annoté. Chaque ligne doit contenir :

```json
{"seed":"...","candidate_a_features":[0,0,0,0,0],"candidate_b_features":[0,0,0,0,0],"label":1}
```

`label=1` signifie que A présente le meilleur compromis progression, sécurité
et stabilité. Le code ne génère jamais ces labels artificiellement. Sans ce
jeu annoté, E reste visible mais son lancement est bloqué.

## 5. Tests automatisés 1 à 9 et provenance des données

```bash
bash scripts/test_report_dreamer.sh
```

Le script compile, exécute les neuf tests progressifs ainsi qu'un test de
provenance native/vérité terrain, entraîne un minuscule candidat diagnostique
et l'évalue sur son split test. Les artefacts sous `/tmp` sont des smoke tests
et ne sont jamais utilisés par la webapp.

## 6. Test 5 CARLA en shadow

Avec un candidat explicite, sans lui donner le contrôle :

```bash
REPORT_DREAMER_ABLATION=D REPORT_DREAMER_SHADOW=1 \
REPORT_DREAMER_CHECKPOINT=$PWD/checkpoints/report_aligned_dreamer/candidate/report_dreamer_candidate.pt \
ROUTE_ID=57 SEED=20260818 \
  bash scripts/run_report_dreamer_live_test.sh
```

Le panneau Pygame expose candidat, alpha, risque/progression prédits et commande
native/proposée/finale. En shadow, la commande finale doit rester native.

La preuve machine de cette invariance se lance ensuite avec :

```bash
~/miniconda3/envs/simlingo/bin/python \
  scripts/verify_report_dreamer_shadow_trace.py \
  --trace logs/report_dreamer_runtime/ID/trace.jsonl
```

Le vérificateur refuse tout tick où `shadow != true`, `alpha != 0`,
`applied != false`, où l'action finale diffère bit à bit de l'action SimLingo,
ou si une observation/prédiction RSSM n'est pas finie. Ce test prouve le
câblage et l'absence d'effet sur le contrôle ; il ne prouve pas une amélioration
de conduite.

## 7. Campagne CARLA appariée A/D

```bash
~/miniconda3/envs/simlingo/bin/python scripts/run_report_dreamer_ab_campaign.py \
  --checkpoint checkpoints/report_aligned_dreamer/candidate/report_dreamer_candidate.pt \
  --ablation D \
  --routes 55,57 \
  --seeds 20260818,20260819,20260820
```

Pour chaque triplet route/seed/météo, A puis D sont exécutés avec les mêmes
conditions. Chaque résultat et log est copié immédiatement dans le dossier de
campagne. Un run sans résultat Bench2Drive frais est exclu avec sa raison, pas
transformé en zéro silencieux.

## 8. Promotion explicite

```bash
~/miniconda3/envs/simlingo/bin/python scripts/promote_report_dreamer_checkpoint.py \
  --candidate checkpoints/report_aligned_dreamer/candidate/report_dreamer_candidate.pt \
  --prediction-metrics checkpoints/report_aligned_dreamer/candidate/test_prediction_metrics.json \
  --closed-loop-summary logs/report_dreamer_campaigns/ID/closed_loop_ab_summary.json
```

La promotion exige d'abord que le modèle réagisse suffisamment aux actions sur
le split test, que le freinage améliore le risque et la collision prédits dans
les états dangereux, puis au moins trois paires CARLA complètes, aucune
régression collision ou off-road et un Driving Score moyen strictement
supérieur à A. La webapp ne charge que le checkpoint promu sous
`checkpoints/report_aligned_dreamer/production/`.

## 9. Traces et KPI

Chaque tick RSSM conserve observation nommée, contexte CARLA, action SimLingo,
candidats, utilités, prédictions, alpha, action finale et latence. Les critères
collision/off-road/règles viennent du résultat Bench2Drive. Ils peuvent être
joints après le run :

```bash
~/miniconda3/envs/simlingo/bin/python scripts/summarize_report_dreamer_run.py \
  --trace logs/report_dreamer_runtime/ID/trace.jsonl \
  --result logs/simlingo_eval/results_ROUTE_seed_SEED.json
```

Un résultat Bench2Drive incomplet conserve également ses métriques à `null` et
une raison d'exclusion ; il n'entre pas dans les agrégats A/D. Les collisions
et sorties de route par tick restent `null` si aucun événement synchronisé
n'est disponible. Les seules mesures contrefactuelles directement
identifiables dans la trace actuelle sont le nombre moyen de candidats
imaginés par décision, l'entropie des utilités candidates, les taux de
proposition et d'intervention, alpha, la latence et les deltas de
risque/progression **prédits par le modèle**. La cohérence avec le futur
réellement observé, le rejet explicite des candidats dangereux et le DQI
restent `N/A` faute de labels synchronisés suffisants. Le dashboard ne les
reconstruit jamais à partir de noms de manoeuvres ou du nombre global
d'incidents.

Le résumé distingue `proposal_rate` (le modèle préfère un candidat non natif)
de `intervention_rate` (alpha non nul et action effectivement appliquée). En
shadow, le premier peut être non nul mais le second doit être exactement nul.

## État scientifique

- Tests 1 à 9 et provenance des données : automatisés.
- Test 5 : intégration shadow exécutée le 21 août 2026 sur Town12, route 57,
  scénario `CrossingBicycleFlow`, seed `20260818`. Les 338 ticks ont conservé
  `alpha=0`, `applied=false` et une action finale strictement identique à
  SimLingo. Le run a été interrompu manuellement après la couverture du test ;
  il ne constitue donc ni un score Bench2Drive final ni une preuve de gain.
- Test 10 : en attente d'un checkpoint de production promu. Il devra être une
  vraie boucle fermée CARLA avec alpha actif et un résultat Bench2Drive
  `Finished`; le candidat diagnostique sous `/tmp` n'est pas admissible.
- Candidat smoke : preuve de fonctionnement logiciel uniquement.
- Checkpoint de production : absent tant que la campagne figée ne valide pas
  l'amélioration.
- Pairwise E : absent tant que des préférences humaines ou expert réellement
  annotées ne sont pas fournies.

La phrase du rapport reste donc exacte : SimLingo est le conducteur de
référence, le RSSM imagine plusieurs conséquences, actor/critic apprennent dans
ces futurs et une autorité continue module l'assistance Dreamer.
