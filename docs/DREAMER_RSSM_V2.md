# Dreamer PPO RSSM V2

## Statut

Cette variante est un candidat experimental, complementaire a SimLingo. Elle ne
remplace ni SimLingo, ni les checkpoints PPO/SDBS actuellement proteges.

- Mode dashboard : `Dreamer PPO RSSM V2 - experimental`
- Checkpoint candidat :
  `external/simlingo/checkpoints/dreamer_ppo_rssm_v2/candidate_model.pt`
- SHA-256 candidat :
  `3d17ef583c92430eed4af520bcf2aaaefdcf68a1d22f0ed6ce94b9c7bb6b4dc8`
- Parent valide avant calibration locale de la tete de risque :
  `candidate_model_before_stationary_risk_head_20260812.pt`
- SHA-256 du parent valide :
  `382b94a0ce592b4177a990faaeed50e5324b99c3b92a5b96e8ecefa56967b80c`
- Tentative trop conservatrice, archivee mais non active :
  `candidate_model_before_conservative_stationary_risk_20260812.pt`
- SHA-256 de la tentative trop conservatrice :
  `71162634c4d96def7ff9fd1c48c41dafdc60e0382e64e3b3b4409447a9831330`
- Sauvegarde exacte avant correction du 12 aout :
  `candidate_model_before_stationary_oncoming_fix_20260812.pt`
- SHA-256 de la sauvegarde :
  `cc7a83e0d170c26b20b0c286d7eea79bee1b482e271bd586c2876232e0596c0c`
- Checkpoint PPO source :
  `external/simlingo/checkpoints/dreamer_ppo_rl_noguard/production_model.pt`
- SHA-256 source :
  `7e183b0d856e251d7a1f5ee04525acb57daa779704e6084b450ecc05f823f3fa`

Le checkpoint source n'est jamais modifie par l'entrainement V2.

## Pourquoi cette migration

L'ancien world model etait un MLP a un pas : il recevait un etat et une action,
puis predisait directement l'etat suivant. Il n'avait ni memoire recurrente, ni
etat latent stochastique, ni apprentissage explicite sur plusieurs pas. Cette
architecture est insuffisante pour distinguer une attente justifiee, un vehicule
en approche et un creneau de depassement qui va rester libre.

La V2 introduit un RSSM categoriel inspire de DreamerV3 :

- encodeur d'observation et d'action ;
- etat recurrent deterministe par GRU ;
- etat latent categoriel stochastique ;
- prior et posterior avec KL dynamique/representation ;
- tetes separees pour delta d'observation, reward, continuation, risque,
  progression et evenements de securite ;
- apprentissage multi-pas par overshooting ;
- imagination a cinq pas au runtime.

Le risque reste distinct du reward, dans l'esprit de SafeDreamer. Nous
n'importons toutefois pas SafeDreamer ou DreamerV3 tels quels : il s'agit d'une
adaptation PyTorch compacte aux traces et au runtime SimLingo existants.

References primaires :

- DreamerV3 : https://arxiv.org/abs/2301.04104
- Configuration officielle DreamerV3 :
  https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml
- SafeDreamer : https://arxiv.org/abs/2307.07176
- Implementation officielle SafeDreamer :
  https://github.com/PKU-Alignment/SafeDreamer
- TD-MPC2, evalue mais non retenu pour cette migration minimale :
  https://github.com/nicklashansen/tdmpc2

## Role au runtime

SimLingo reste le conducteur principal. A chaque tick, la V2 compare dans le
world model deux candidats :

1. la commande native SimLingo ;
2. la proposition de la policy PPO migree.

Chaque candidat est imagine sur cinq pas. Comme le Dreamer reste un complement,
sa proposition n'est engagee que sur le premier pas imagine ; les quatre pas
suivants reprennent la commande SimLingo. Cela evite de valoriser artificiellement
un braquage Dreamer repete en boucle comme s'il etait un controleur autonome.

Le score continu combine progression predite, risque futur convexe et amplitude
de l'ecart avec SimLingo. La courbure du risque et la penalite d'action sont
calibrees depuis les erreurs du split de validation. La commande PPO n'est
appliquee que si son utilite imaginee est meilleure ; sinon la commande SimLingo
est conservee.

Ce chemin V2 n'utilise aucun turn guard, collision shield, seuil TTC ou regle de
clearance. Les coefficients de reward/risque sont les poids de l'objectif appris,
pas des regles de conduite codees a la main.

## Correction des vehicules opposes immobiles

La collecte historique marquait un vehicule comme `oncoming` seulement si son
orientation et sa vitesse instantanee indiquaient toutes deux un rapprochement.
Un vehicule oppose arrete au feu ou dans un embouteillage disparaissait donc de
l'observation exactement au moment ou le RSSM devait evaluer un depassement.

La representation corrigee separe maintenant les deux faits :

- l'orientation opposee est determinee par le produit scalaire des headings ;
- la vitesse de rapprochement sert au TTC, mais pas a l'existence du vehicule ;
- un vehicule oppose immobile conserve une distance finie et un TTC non urgent ;
- les anciennes traces dont le booleen etait faux sont reparees par geometrie ;
- les slots temporels de trafic du vecteur 49D sont reconstruits, meme lorsqu'un
  ancien `policy_state_vector` etait deja stocke dans la trace.

Ce changement est une correction d'etat, pas un guard : aucune distance ne
declenche directement un freinage, un veto ou un depassement.

## Donnees et entrainement actuel

- 34 traces ordonnees ;
- 37 episodes ;
- 18 684 transitions ;
- 867 fenetres sequentielles ;
- routes de validation tenues a l'ecart : 32, 33, 36 et 55 ;
- 2 068 transitions enseignant disponibles pour controler la migration PPO.

Pour le recalibrage du 12 aout, deux traces supplementaires ont ete exclues du
pool d'apprentissage et reservees a la validation de securite :

- Town13, route 70, seed 451052 : attente bloquee avec trafic oppose ;
- Town10HD, route 148, seed 208080 : depassement termine a 100 %, sans collision
  ni off-road.

Commande reproductible :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python scripts/train_dreamer_rssm_v2.py
```

Le script ecrit toujours `last_attempt.pt` et
`validation_report.json`. Il ne remplace `candidate_model.pt` que si la quality
gate passe. Il ne touche jamais au checkpoint PPO source.

Le recalibrage d'un checkpoint deja valide se reproduit avec :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python \
  scripts/recalibrate_dreamer_rssm_v2.py --promote \
  --validation-trace-pattern \
  'logs/dreamer_online_rl/webapp_20260811_135402_ppo_route_70_seed_451052/trace.jsonl' \
  --validation-trace-pattern \
  'logs/dreamer_online_rl/webapp_20260807_153143_ppo_route_148_seed_208080/trace.jsonl'
```

Cette commande sauvegarde le checkpoint parent, reevalue ses poids sur les
traces corrigees, puis ne promeut que les metadonnees d'arbitrage si toutes les
gates passent. Elle ne reentraine ni le RSSM ni l'acteur.

La calibration locale de la representation opposee se reproduit ensuite avec :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python \
  scripts/finetune_dreamer_rssm_stationary_oncoming.py --promote
```

Cette etape ne modifie que les six tenseurs de `head_risk`. Les dynamiques RSSM,
la tete de progression, la policy PPO et SimLingo restent geles. La cible de
risque est continue et distille le parent : elle ne contient aucun seuil TTC,
aucune regle de clearance et aucun veto d'action. Le checkpoint actif n'est
remplace que si une validation independante conserve les interventions utiles
sur Town10HD tout en reduisant les interventions non productives sur Town13.

## Resultats hors ligne

La quality gate initiale passait. Apres correction de la representation des
vehicules opposes immobiles, une tentative de reentrainement complet a ete
rejetee (`H5 decision = 0.920`). Les poids precedemment valides ont donc ete
conserves : sur exactement les memes traces corrigees, ils donnent
`H5 decision = 0.214` et `H5 risque = 0.134`.

La gate combinee actuelle passe :

- ratio MAE ego a un pas face a la persistence : `1.188` ;
- MAE normalisee des variables de decision a cinq pas : `0.175` ;
- MAE du risque a cinq pas : `0.134` ;
- MAE du risque a quinze pas : `0.167` ;
- Brier evenements a cinq pas : `0.021` ;
- bruit de decision sur etats inactifs a un pas : `0.017`.

Sur les seules traces Town10HD/Town13 forcees, `H5 risque = 0.134` et le Brier
evenements vaut `0.007`. Les coefficients produits sont `risk_curvature=3.103`,
`action_penalty=0.296`, sans seuil dur. Le rapport complet est dans
`recalibration_report.json`.

Un replay contrefactuel a ensuite compare l'ancien arbitre et l'arbitre
recalibre sur les memes etats et actions enregistres :

- Town13 bloquee : l'action Dreamer etait preferee sur `750/975` pas avec
  l'ancien rollout, contre `43/975` avec le rollout complementaire recalibre ;
- Town10HD reussi : `54/436` propositions Dreamer utiles restent preferees ;
- aucune proposition augmentant le risque predit de plus de `0.02` n'est
  retenue par le nouvel arbitre sur ces deux traces.

Une premiere adaptation de la tete de risque a ensuite ete rejetee parce
qu'elle supprimait presque toutes les interventions, y compris le depassement
Town10HD utile. Elle est conservee uniquement pour l'audit sous le SHA
`71162634...`.

La calibration distillee finalement promue donne, sur le replay exact de la
decision runtime a cinq pas :

- Town13 bloquee : `17/821` propositions materielles choisies par le parent,
  contre `2/821` pour le candidat ;
- Town10HD de preservation : `21/258` propositions utiles choisies par le
  parent et `21/258` conservees par le candidat ;
- proposition retenue avec hausse de risque predite superieure a `0.02` : `0` ;
- acteur PPO modifie : non ;
- parametres modifies : uniquement `head_risk.*` ;
- seuil dur ou guard runtime : aucun.

Le rapport reproductible est
`stationary_oncoming_finetune_report.json`.

Ce replay verifie le sens de la correction, mais ne constitue pas une mesure
closed-loop : les KPI finaux doivent toujours venir de l'ablation CARLA avec
routes et seeds fixes.

Limite importante : la reconstruction globale ne bat pas la persistence sur les
horizons longs, surtout pour le trafic exogene et les commandes futures de
SimLingo. La gate valide donc un modele de planification risque/progression, pas
un simulateur complet du monde. Cette limite doit rester visible dans les
rapports.

L'adaptateur latent de l'acteur PPO a ete rejete par la validation. Les nouvelles
colonnes latentes de la policy sont donc initialisees a zero et la commande PPO
historique est preservee exactement. L'effet V2 vient du RSSM et de l'arbitrage
appris, pas d'une modification non validee de l'acteur.

## Calibration de l'utilite pairwise (17 aout 2026)

Le contexte brut 49-D faisait memoriser les seeds au calibrateur. Il a ete
remplace par 18 observations normalisees et map-invariant : vitesse, commande
SimLingo, blocage, clearances/TTC lateraux, disponibilite des voies et trafic
oppose adjacent/courant. Ces valeurs alimentent un reseau appris continu ; il
n'existe toujours aucun seuil, veto ou guard au runtime.

La selection s'arrete au premier epoch qui passe toutes les gates hors
entrainement. Les deux folds inverses passent desormais :

- fold A : epoch `76`, blend `0.575`, positif `70.7 %`, negatif `99.3 %` ;
- fold B : epoch `46`, blend `0.975`, positif `70.4 %`, negatif `98.2 %` ;
- RSSM physique modifie : non ; acteur PPO modifie : non ;
- SimLingo modifie : non ; hard guard runtime : non.

Le candidat conservateur du fold A est isole dans
`utility_calibrator_candidate_pre_ab.pt` (SHA-256 `2cc45fc2...`). Le checkpoint
actif reste `candidate_model.pt` (SHA-256 `d5800f1f...`) : aucune promotion n'a
eu lieu avant l'A/B ferme.

Le lancement A/B depuis le sandbox de developpement a ete arrete avant que
CARLA soit pret (log evaluateur vide), donc ce run ne constitue ni un succes ni
un echec du modele. La commande locale graphique reproductible est :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python \
  scripts/run_dreamer_rssm_v2_ablation.py \
  --run-id utility_context_v2_closed_loop \
  --modes native,rssm_v2 \
  --routes 148,70 \
  --rssm-checkpoint \
    external/simlingo/checkpoints/dreamer_ppo_rssm_v2/utility_calibrator_candidate_pre_ab.pt \
  --reuse-native-report \
    logs/dreamer_rssm_v2_ablation/continuous_authority_reused_native_20260812/report.json \
  --max-wall-seconds 600 --carla-quality Low --camera chase
```

## Regle de promotion

Le candidat ne peut etre declare meilleur que SimLingo ou que le PPO protege sur
la seule base des metriques hors ligne. Une promotion exige un A/B ferme :

- memes routes XML ;
- memes seeds ;
- memes scenarios et densite de trafic ;
- au moins SimLingo natif, PPO protege et RSSM V2 ;
- plusieurs repetitions par route ;
- route completion, driving score, collisions, off-road, infractions, blocage,
  temps d'attente justifie, taux d'intervention et DQI ;
- aucune promotion si le gain moyen masque une regression de securite.

Tant que cette campagne CARLA/Bench2Drive n'est pas terminee, l'etiquette
`experimental` est obligatoire.

La matrice peut etre controlee sans lancer CARLA :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python \
  scripts/run_dreamer_rssm_v2_ablation.py --dry-run
```

Puis lancee lorsque le pilote NVIDIA et CARLA sont disponibles :

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python \
  scripts/run_dreamer_rssm_v2_ablation.py
```

Le runner effectue 24 evaluations (8 routes x 3 modes), sans entrainement ni
promotion automatique. Les sorties sont archivees dans
`logs/dreamer_rssm_v2_ablation/` avec la matrice, chaque resultat Bench2Drive,
les agregats et la decision de gate.

## Verification logicielle

```bash
cd ~/Desktop/vla-av
/home/mohm/miniconda3/envs/simlingo/bin/python -m unittest \
  tests.test_dreamer_rssm_v2 \
  tests.test_dreamer_online_rl_reward \
  tests.test_dreamer_rl_complement
```

La suite couvre notamment les dimensions sequentielles, le posterior
deterministe, la migration exacte de l'acteur, le chargement runtime, le rollout
multi-pas, le gel des slots statiques et la capacite de l'arbitre a conserver
SimLingo sans hard guard.

### Action shooting sans guard

Le runtime RSSM calibré ne dépend plus uniquement de la moyenne déterministe
de l'acteur. À chaque tick, il construit des sigma-points dans la distribution
continue apprise par PPO (direction, longitudinal et combinaisons), imagine
leurs futurs avec le RSSM, puis conserve l'action ayant la meilleure utilité
risque/progression. Cette recherche n'utilise ni seuil de distance, ni compteur
de blocage, ni veto géométrique. SimLingo reste l'action de référence et peut
être conservé avec une autorité Dreamer exactement nulle.

Le dashboard sélectionne automatiquement
`utility_calibrator_candidate_pre_ab.pt` pour le mode RSSM tout en conservant
`candidate_model.pt` comme retour arrière inchangé. Les traces exposent
`rssm_planner_candidates`, le type du sigma-point choisi et le résidu d'utilité
appris afin de rendre chaque intervention vérifiable.
