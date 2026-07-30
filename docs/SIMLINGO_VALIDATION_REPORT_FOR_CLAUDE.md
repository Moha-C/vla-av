# Rapport de validation SimLingo - VLA-AV

Date de validation locale : 20 mai 2026  
Projet : VLA-AV / CARLA closed-loop autonomous driving  
Objectif : documenter l'integration de SimLingo comme baseline VLA fonctionnelle dans notre pipeline, avec les tests deja executes et les limites restantes.

## Resume executif

SimLingo est maintenant integre dans le pipeline local VLA-AV comme baseline principale de conduite autonome closed-loop sous CARLA/Bench2Drive.

Le systeme permet :

- de lancer SimLingo depuis une interface web locale ;
- de choisir une map, une route, un scenario, une seed, la qualite CARLA et le mode de camera ;
- d'afficher la simulation via Pygame en POV `chase`, `wheel`, `front` ou `top` ;
- d'executer SimLingo comme agent autonome closed-loop qui controle effectivement le vehicule via CARLA ;
- de recuperer les resultats Bench2Drive dans `logs/simlingo_eval/results_*.json`.

La validation montre que SimLingo fonctionne correctement comme agent autonome sur plusieurs routes representatives. Plusieurs tests terminent la route a 100% sans collision, sans feu rouge, sans sortie de route et sans blocage agent. En revanche, SimLingo n'est pas parfait sur tous les scenarios : un test feu/jonction a termine la route a 100% mais avec une collision vehicule, et les routes longues custom restent experimentales.

Conclusion : SimLingo est valide comme baseline VLA fonctionnelle et implementee dans notre pipeline, mais pas comme modele infaillible sur toutes les routes.

## Positionnement technique

SimLingo est un modele VLA/VLM pour conduite autonome en simulation CARLA. Il combine :

- perception camera ;
- backbone vision-language, notamment InternVL2-1B dans notre checkpoint ;
- conditionnement par route / target points Bench2Drive ;
- prediction de waypoints de trajectoire et de vitesse ;
- conversion des predictions en commandes CARLA `steer`, `throttle`, `brake`.

Dans notre pipeline, SimLingo est utilise comme agent closed-loop complet : a chaque tick CARLA, il recoit les observations et le contexte de route, predit l'action, puis controle le vehicule. Ce n'est donc pas seulement un VLM descriptif : il agit comme conducteur autonome dans CARLA.

Pour les demos locales, nous utilisons un mode rapide :

- `SIMLINGO_FAST_DRIVING=1`
- `SIMLINGO_MODEL_EVERY_N=5`
- `SIMLINGO_SKIP_JPEG=1`
- `SIMLINGO_TURN_SPEED_GUARD=1`

Ce mode conserve la conduite closed-loop par waypoints, mais reduit le cout de generation langage frame par frame pour rendre la demonstration plus fluide.

## Pipeline implemente

Pipeline utilisateur :

1. L'utilisateur ouvre le dashboard local.
2. Il choisit map, route, scenario, seed, qualite CARLA, mode POV et options de conduite.
3. Le dashboard lance `scripts/run_simlingo_with_pov.sh`.
4. Le script lance d'abord le viewer Pygame.
5. Le script lance ensuite `scripts/run_simlingo_local_eval.sh`.
6. Bench2Drive charge la route XML.
7. `team_code/agent_simlingo.py` charge le checkpoint SimLingo.
8. CARLA execute la simulation closed-loop.
9. Les resultats sont sauvegardes dans `logs/simlingo_eval/`.

Fichiers principaux :

- `scripts/simlingo_dashboard.py` : dashboard local de parametrage et lancement.
- `scripts/run_simlingo_dashboard.sh` : demarrage du dashboard, avec choix automatique d'un port libre.
- `scripts/run_simlingo_with_pov.sh` : lancement SimLingo + Pygame POV.
- `scripts/run_simlingo_local_eval.sh` : lancement CARLA + evaluator Bench2Drive.
- `scripts/carla_ego_viewer.py` : viewer Pygame chase/wheel/front/top.
- `external/simlingo/team_code/agent_simlingo.py` : agent SimLingo adapte pour le mode demo rapide.
- `scripts/simlingo_route_healthcheck.py` : verification du catalogue routes/maps.

## Adaptations ajoutees au projet

### Dashboard local

Le dashboard permet de lancer SimLingo sans retaper les commandes shell a chaque fois. Il expose toutes les routes Bench2Drive installees, pas seulement Town12 et Town13.

Validation API dashboard :

- test local sur port temporaire `8777` ;
- endpoint teste : `/api/routes` ;
- resultat : `routes=220` ;
- resultat : `show_experimental=True` ;
- towns detectees : `Town01`, `Town02`, `Town03`, `Town04`, `Town05`, `Town06`, `Town07`, `Town10HD`, `Town11`, `Town12`, `Town13`, `Town15` et variantes `_Opt` installees cote CARLA.

### Viewer Pygame

Le viewer Pygame a ete stabilise et ameliore :

- camera `chase` ;
- camera `wheel` ;
- camera `front` ;
- camera `top` ;
- qualite 1280x720 par defaut ;
- FOV reglable ;
- contraste / luminosite / saturation ;
- filtre anti-frames noires ;
- conservation de la derniere bonne frame si une frame noire est recue ;
- option `--disable-black-frame-filter` pour debug.

Parametres anti-frame noire ajoutes, avec seuils par defaut renforces :

- `SIMLINGO_VIEW_MIN_BRIGHTNESS=12`
- `SIMLINGO_VIEW_MIN_P95=24`
- `SIMLINGO_VIEW_DARK_DROP_RATIO=0.45`
- `SIMLINGO_VIEW_STALE_SECONDS=60`

### Stabilisation conduite demo

Le mode demo rapide ajoute :

- cache de prediction pour eviter une inference VLM lourde a chaque tick ;
- skip de generation JPEG/langage par frame ;
- profilage regulier des temps de tick, inference, controle et vitesse ;
- garde de vitesse en virage (`SIMLINGO_TURN_SPEED_GUARD`) pour limiter les zigzags et offroad lors des virages trop rapides ;
- logs de diagnostic `SIMLINGO_PROFILE` et `SIMLINGO_TURN_GUARD`.

## Tests statiques executes

### Compilation Python

Commande executee :

```bash
python3 -m py_compile \
  scripts/simlingo_dashboard.py \
  scripts/carla_ego_viewer.py \
  scripts/simlingo_route_healthcheck.py \
  external/simlingo/team_code/agent_simlingo.py
```

Resultat : succes.

### Syntaxe Bash

Commande executee :

```bash
bash -n \
  scripts/run_simlingo_with_pov.sh \
  scripts/run_simlingo_dashboard.sh \
  scripts/run_simlingo_local_eval.sh \
  scripts/generate_simlingo_long_route.sh \
  scripts/run_simlingo_long_route_with_pov.sh
```

Resultat : succes.

### Catalogue routes/maps

Commande executee :

```bash
python3 scripts/simlingo_route_healthcheck.py
```

Resultat global :

- `dashboard_enabled_routes=220`
- toutes les routes Bench2Drive installees sont visibles par le dashboard.

Couverture par town :

| Town | Routes exposees | VRU | Feux | Stop | Jonctions |
|---|---:|---:|---:|---:|---:|
| Town01 | 4 | 1 | 0 | 0 | 1 |
| Town02 | 4 | 2 | 0 | 0 | 1 |
| Town03 | 11 | 2 | 1 | 0 | 1 |
| Town04 | 12 | 1 | 2 | 0 | 4 |
| Town05 | 9 | 0 | 2 | 0 | 2 |
| Town06 | 6 | 0 | 0 | 0 | 0 |
| Town07 | 5 | 1 | 2 | 0 | 1 |
| Town10HD | 4 | 0 | 0 | 0 | 1 |
| Town11 | 7 | 2 | 0 | 0 | 0 |
| Town12 | 104 | 11 | 18 | 3 | 19 |
| Town13 | 47 | 4 | 3 | 2 | 3 |
| Town15 | 7 | 1 | 2 | 0 | 2 |

## Tests closed-loop CARLA executes

Les tests suivants ont ete executes localement en CARLA offscreen avec SimLingo comme agent autonome.

Configuration commune :

- CARLA 0.9.15 ;
- Bench2Drive routes XML ;
- checkpoint SimLingo officiel dans `models/simlingo_hf/`;
- mode rapide demo active ;
- guard de vitesse en virage active ;
- rendu offscreen pour les tests automatises.

### Test 1 - VRU / bicycle crossing

Route : `bench2drive_55.xml`  
Town : `Town12`  
Scenario : `CrossingBicycleFlow_1`  
Seed : `217613`  
Fichier resultat : `logs/simlingo_eval/results_bench2drive_55_seed_217613.json`

Resultats :

- status : `Completed`
- route completion : `100%`
- driving score : `100.0`
- infraction penalty : `1.0`
- collision pieton : `0`
- collision vehicule : `0`
- collision layout : `0`
- feu rouge : `0`
- stop : `0`
- outside route lanes : `0`
- route deviation : `0`
- agent blocked : `0`
- timeout : `0`
- duree game : `14.15s`
- duree system : `23.075s`

Conclusion : test reussi cote conduite et securite. `MinSpeedTest` signale des ecarts de vitesse, mais sans penaliser le score compose.

### Test 2 - VRU / pedestrian crossing

Route : `bench2drive_64.xml`  
Town : `Town13`  
Scenario : `ParkingCrossingPedestrian_1`  
Seed : `317717`  
Fichier resultat : `logs/simlingo_eval/results_bench2drive_64_seed_317717.json`

Resultats :

- status : `Completed`
- route completion : `100%`
- driving score : `100.0`
- infraction penalty : `1.0`
- collision pieton : `0`
- collision vehicule : `0`
- collision layout : `0`
- feu rouge : `0`
- stop : `0`
- outside route lanes : `0`
- route deviation : `0`
- agent blocked : `0`
- timeout : `0`
- duree game : `34.1s`
- duree system : `43.055s`

Conclusion : test reussi cote conduite et securite.

### Test 3 - Feu / jonction

Route : `bench2drive_08.xml`  
Town : `Town12`  
Scenario : `SignalizedJunctionRightTurn_1`  
Seed : `420801`  
Fichier resultat : `logs/simlingo_eval/results_bench2drive_08_seed_420801.json`

Resultats :

- status : `Completed`
- route completion : `100%`
- driving score : `60.0`
- infraction penalty : `0.6`
- collision vehicule : `1`
- feu rouge : `0`
- stop : `0`
- outside route lanes : `0`
- route deviation : `0`
- agent blocked : `0`
- timeout : `0`
- duree game : `13.85s`
- duree system : `18.021s`

Conclusion : SimLingo a bien termine la route et respecte le feu, mais collision vehicule detectee. Ce test montre une limite reelle du modele sur certains scenarios dynamiques.

### Test 4 - Stop / jonction non signalisee

Route : `bench2drive_29.xml`  
Town : `Town12`  
Scenario : `VanillaNonSignalizedTurnEncounterStopsign_1`  
Seed : `420829`  
Fichier resultat : `logs/simlingo_eval/results_bench2drive_29_seed_420829.json`

Resultats :

- status : `Completed`
- route completion : `100%`
- driving score : `100.0`
- infraction penalty : `1.0`
- collision pieton : `0`
- collision vehicule : `0`
- collision layout : `0`
- feu rouge : `0`
- stop : `0`
- outside route lanes : `0`
- route deviation : `0`
- agent blocked : `0`
- timeout : `0`
- duree game : `10.1s`
- duree system : `13.013s`

Conclusion : test reussi cote conduite et securite.

## Resultats historiques disponibles

En plus des tests frais ci-dessus, le dossier `logs/simlingo_eval/` contient 17 records Bench2Drive parses.

Synthese :

- records parses : `17`
- towns couvertes par les resultats disponibles : `Town12`, `Town13`, `Town01`
- scenarios observes : `CrossingBicycleFlow`, `ParkingCrossingPedestrian`, `PedestrianCrossing`, `VehicleTurningRoutePedestrian`, `ParkingCutIn`, `StaticCutIn`, `SignalizedJunctionRightTurn`, `NonSignalizedJunctionLeftTurnEnterFlow`, `HardBreakRoute`, `T_Junction`, `NoScenario`
- routes terminees : `10`
- routes terminees a 100% sans infraction securite majeure : `9`

Les anciens crashs `tuple indices must be integers or slices, not tuple` appartiennent a des runs avant correctifs. Les tests frais apres correctifs ne reproduisent pas ce crash.

## Interpretation du MinSpeedTest

Bench2Drive affiche souvent `MinSpeedTest FAILURE`, meme lorsque :

- route completion = `100%` ;
- collision = `0` ;
- red light = `0` ;
- stop infraction = `0` ;
- outside route lanes = `0` ;
- driving score = `100.0`.

Dans les JSON, ces infractions apparaissent sous forme :

```text
Average speed is X% of the surrounding traffic's one
```

Cela indique un ecart de vitesse relative par rapport au trafic environnant. Ce n'est pas toujours une erreur bloquante de conduite. Dans plusieurs tests, le score compose reste `100.0` malgre `MinSpeedTest FAILURE`.

Pour la documentation, il faut donc distinguer :

- criteres de securite : collision, feu rouge, stop, offroad, blocage ;
- critere de regularite vitesse : `MinSpeedTest`.

## Limites identifiees

### 1. SimLingo n'est pas parfait sur tous les scenarios

Le test `SignalizedJunctionRightTurn_1` a termine la route a 100%, mais avec une collision vehicule. Cela confirme que SimLingo est une baseline fonctionnelle, pas une garantie de conduite parfaite.

### 2. Les routes longues custom restent experimentales

Nous avons genere des routes longues custom en combinant des routes Bench2Drive existantes. Ces routes peuvent creer des transitions geometriquement difficiles ou trop abruptes. Un run long route Town12 a produit offroad/collision/red-light apres une longue sequence.

Conclusion : pour les resultats officiels, il faut privilegier les routes Bench2Drive originales. Les routes longues sont utiles pour demo/recherche, mais pas encore comme benchmark robuste.

### 3. Evaluation langage non executee localement

SimLingo inclut aussi des capacites VQA, commentary et Action Dreaming selon le papier et le repo. Dans cette validation locale, nous avons valide principalement la conduite closed-loop CARLA.

Les evaluations langage completes necessitent :

- les datasets SimLingo correspondants ;
- les scripts d'evaluation `simlingo_training/eval.py` ;
- possiblement une cle API selon le mode d'evaluation des reponses langage.

Ces tests restent a faire si l'objectif est de documenter toute la partie vision-language au meme niveau que la conduite.

### 4. Evaluation officielle 220 routes non executee

Le dashboard expose bien les 220 routes installees, mais nous n'avons pas lance une evaluation exhaustive des 220 routes pendant cette session. Une telle evaluation prendrait plusieurs heures et doit etre planifiee comme benchmark dedie.

## Commandes de reproduction

Lancer le dashboard :

```bash
cd ~/Desktop/vla-av
bash scripts/run_simlingo_dashboard.sh
```

Verifier le catalogue routes/maps :

```bash
cd ~/Desktop/vla-av
python3 scripts/simlingo_route_healthcheck.py
```

Lancer un test VRU propre :

```bash
cd ~/Desktop/vla-av
ROUTE_ID=55 SEED=217613 PORT=2000 TM_PORT=8000 \
CARLA_QUALITY=Low SIMLINGO_RENDER_MODE=offscreen \
SIMLINGO_FAST_DRIVING=1 SIMLINGO_MODEL_EVERY_N=5 SIMLINGO_TURN_SPEED_GUARD=1 \
bash scripts/run_simlingo_local_eval.sh
```

Lancer une simulation visible avec Pygame :

```bash
cd ~/Desktop/vla-av
ROUTE_ID=55 SEED=$RANDOM PORT=2000 TM_PORT=8000 \
SIMLINGO_VIEW_MODE=chase SIMLINGO_VISUAL_WEATHER=day CARLA_QUALITY=Epic \
bash scripts/run_simlingo_with_pov.sh
```

Regler plus agressivement les frames noires si besoin :

```bash
cd ~/Desktop/vla-av
SIMLINGO_VIEW_MIN_BRIGHTNESS=12 \
SIMLINGO_VIEW_MIN_P95=24 \
SIMLINGO_VIEW_DARK_DROP_RATIO=0.45 \
SIMLINGO_VIEW_STALE_SECONDS=120 \
ROUTE_ID=55 SEED=$RANDOM PORT=2000 TM_PORT=8000 \
SIMLINGO_VIEW_MODE=chase SIMLINGO_VISUAL_WEATHER=day CARLA_QUALITY=Epic \
bash scripts/run_simlingo_with_pov.sh
```

## Recommandation pour la suite

SimLingo doit etre presente comme :

> une baseline VLA open-source closed-loop adaptee dans notre pipeline VLA-AV pour lancer, visualiser et evaluer des scenarios CARLA/Bench2Drive.

Les contributions projet a mettre en avant :

- integration locale complete ;
- dashboard de controle ;
- viewer Pygame stabilise ;
- gestion des routes/maps/scenarios ;
- logs Bench2Drive exploitables ;
- adaptation mode demo rapide ;
- garde de vitesse en virage ;
- preparation a l'integration future avec les attaques SUMO/red-team.

La formulation honnete est :

> Nous n'avons pas cree le checkpoint SimLingo original. Nous avons integre et adapte SimLingo dans notre propre plateforme VLA-AV, avec une interface de controle, un viewer Pygame, des options de scenario, des scripts de validation et une base de tests reproductibles. SimLingo devient ainsi notre baseline autonome closed-loop pour la suite du projet.

## Conclusion finale

SimLingo est fonctionnel dans notre pipeline local :

- environnement installe ;
- checkpoint charge ;
- CARLA lance ;
- routes Bench2Drive chargees ;
- vehicule controle automatiquement par le VLA ;
- dashboard operationnel ;
- Pygame operationnel ;
- 220 routes exposees ;
- plusieurs tests closed-loop reussis avec route completion 100% et zero infraction securite ;
- limites documentees sur certains scenarios et sur les routes longues custom.

Le projet peut donc officiellement passer a la phase suivante :

1. utiliser SimLingo comme baseline principale ;
2. accumuler des resultats Bench2Drive propres ;
3. ameliorer les scenarios custom ;
4. integrer progressivement les perturbations SUMO/red-team ;
5. garder Alpamayo-R1 comme piste secondaire ou comparative.
