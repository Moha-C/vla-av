# Resume complet depuis l'integration de SimLingo

Date de synthese : 2026-08-04
Projet actif : `/home/mohm/Desktop/vla-av`

Ce document resume l'evolution du projet depuis l'integration de SimLingo dans
notre pipeline CARLA/Bench2Drive, avec un regard critique sur ce qui a ete garde,
mis de cote, corrige ou transforme.

## 1. Objectif initial autour de SimLingo

L'objectif de depart etait d'integrer SimLingo comme modele VLA de reference
dans CARLA, puis de l'utiliser comme base pour construire une plateforme de
test, de demonstration et d'amelioration.

La logique retenue tres tot a ete la suivante :

- SimLingo natif doit rester la baseline scientifique.
- Toute amelioration doit etre comparee a cette baseline sur les memes routes,
  les memes maps et les memes scenarios Bench2Drive.
- Les ajouts externes ne doivent pas etre presentes comme "SimLingo pur" s'ils
  modifient le controle ou introduisent des heuristiques.
- Le pipeline doit etre pilotable depuis une interface web locale, pas seulement
  par commandes shell.

## 2. Integration SimLingo / CARLA / Bench2Drive

SimLingo a ete integre dans le dossier `external/simlingo`, avec ses chemins
Bench2Drive, son agent `team_code/agent_simlingo.py`, son checkpoint Hugging Face
et ses routes XML natives.

Les points importants mis en place :

- lancement closed-loop CARLA depuis scripts shell;
- chargement du checkpoint SimLingo :
  `models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`;
- support des routes Bench2Drive XML natives;
- lecture automatique de la town et du scenario depuis les XML;
- gestion des logs dans `logs/simlingo_eval`;
- resultats Bench2Drive en JSON avec route completion, collisions, offroad,
  red light, blocked agent, timeout, etc.

La premiere grosse etape a donc ete de rendre SimLingo executable en closed-loop,
avec CARLA lance automatiquement et les resultats recuperables.

## 3. Probleme de visualisation et Pygame POV

Au debut, la demonstration etait difficilement exploitable :

- ecran noir ou frames noires entre deux frames;
- fenetre Pygame de mauvaise qualite;
- ralentissement important;
- pas de visualisation claire des decisions;
- necessite de relancer l'interface apres crash/stop.

On a donc ajoute et stabilise un viewer Pygame :

- vue chase, front, wheel et top;
- qualite CARLA/Pygame en Epic 1080p;
- correction partielle des frames noires;
- affichage de la vitesse;
- overlay des trajectoires :
  - rouge : predicted path waypoints;
  - vert : predicted speed waypoints;
  - bleu : target points;
- enregistrement video automatique de la fenetre Pygame;
- replay accelere apres simulation, avec choix x3, x5, x10, x50, x100, etc.

Decision critique : la simulation elle-meme ne doit pas etre artificiellement
acceleree si on veut evaluer SimLingo proprement. Le replay peut etre accelere
apres coup, mais le run closed-loop doit rester natif.

## 4. Suppression des hacks de conduite rapide / safety mode

Au debut, des options de demonstration rapide existaient :

- `SIMLINGO_FAST_DRIVING`;
- cache VLA toutes les N frames;
- `turn_guard`;
- unstuck artificiel;
- limitation ou correction de steering;
- post-process de controle.

Ces options rendaient la demo plus fluide, mais elles posaient un probleme
scientifique : le vehicule n'etait plus 100% SimLingo natif.

Decision retenue :

- supprimer ou desactiver les modes safety/fast qui modifient la conduite native;
- garder uniquement les options de visualisation;
- laisser SimLingo recalculer naturellement sa prediction;
- accepter une demo lente, puis rejouer la video en accelere.

Critique : c'etait necessaire pour eviter de confondre "amelioration SimLingo"
et "autopilot cache". Cela a clarifie toute la suite du projet.

## 5. Routes, maps et scenarios Bench2Drive

On a d'abord filtre certaines maps parce que seules Town12/Town13 semblaient
fonctionner correctement. Ensuite, il a ete decide de ne pas masquer les autres
maps natives.

Objectif final :

- exposer le maximum de routes natives SimLingo/Bench2Drive;
- ne pas inventer de scenarios;
- utiliser ce qui existe dans le repo SimLingo et Bench2Drive;
- garder les routes XML natives comme source de verite.

Les categories de scenarios utilisees dans l'interface :

- VRU / crossing;
- traffic light;
- stop;
- junction;
- accident;
- cut-in / parking;
- actor flow;
- all native scenarios.

Observation critique : certaines routes/scenarios reussissent tres bien,
d'autres peuvent echouer meme en SimLingo natif. Ce n'est pas forcement une
erreur du pipeline : cela peut etre une limite du modele ou une difficulte de
Bench2Drive.

## 6. Routes longues et custom map

On a essaye de generer des routes longues en assemblant des keypoints depuis les
XML Bench2Drive.

Ce qui a marche :

- generation offline de longues routes XML;
- lancement dans Town12;
- conduite possible pendant un certain temps.

Ce qui n'a pas ete satisfaisant :

- certaines routes n'avaient pas de scenario valide;
- erreurs de type `scenario_configs[0]`;
- raccords de keypoints parfois instables;
- zigzags, offroad et perte de controle sur longues routes;
- benchmark moins defendable car la route n'est plus un scenario natif propre.

Decision :

- garder les routes custom comme outil exploratoire/demo;
- ne pas les utiliser comme base principale de comparaison scientifique.

## 7. Interface web SimLingo

Une webapp locale a ete construite pour piloter le projet.

Elle permet de choisir :

- map/town;
- scenario;
- route XML;
- seed;
- qualite CARLA;
- POV Pygame;
- replay speed;
- mode CARLA POV;
- mode CARLA + SUMO mirror;
- Dreamer mode;
- CoT externe;
- TwinSentinel attacks;
- SAFE-DREAM KPI.

L'interface a aussi ete retravaillee visuellement pour donner une impression de
"world" CARLA/SimLingo, plus proche d'un cockpit de demonstration que d'un simple
formulaire.

Ameliorations importantes :

- un seul bouton `Launch` qui prend en compte les options selectionnees;
- `Stop` qui n'oblige plus a relancer le dashboard;
- maintien du localhost apres arret d'une simulation;
- nettoyage progressif des modes obsoletes;
- meilleure lisibilite des KPI.

Critique : l'interface est devenue le centre du pipeline. Cela simplifie les
tests, mais impose de bien separer les modes experimentaux des modes stables.

## 8. Baseline SimLingo et scenarios accident

Plusieurs tests ont montre que SimLingo natif pouvait parfois rester bloque
derriere un accident, notamment sur certains scenarios Town10HD.

Observation importante :

- SimLingo peut predire une trajectoire devant lui mais ne pas prendre
  l'initiative de depasser;
- il peut suivre le target point global sans vraiment raisonner sur le
  contournement;
- dans certains cas, il reste a 0 km/h derriere l'obstacle.

Cela a motive l'axe Dreamer : non pas remplacer SimLingo, mais lui ajouter un
module capable d'evaluer des actions alternatives quand SimLingo bloque.

## 9. Trois axes de recherche retenus

Trois axes ont ete selectionnes pour tenter d'outperform SimLingo :

1. CoT utile pour la conduite dangereuse.
2. Action Dreaming custom.
3. Red-team SUMO / attaques trafic.

Critique des axes :

- CoT : prometteur pour l'explicabilite, mais pas suffisant seul pour ameliorer
  le controle.
- Action Dreaming : axe le plus proche de SimLingo, exploitable pour generer des
  donnees et entrainer un module complementaire.
- SUMO attacks : axe tres fort scientifiquement car il teste la robustesse sous
  perturbation dynamique de l'environnement.

## 10. Action Dreaming et clarification du "Dreamer"

Une confusion initiale existait autour d'Action Dreaming :

- dans SimLingo, Action Dreaming est surtout une strategie de generation de
  supervision/instructions;
- ce n'est pas automatiquement un vrai world model Dreamer utilise en closed-loop;
- pour avoir un "vrai Dreamer", il faut un module qui predit des futurs, score
  des actions et influence le controle.

Decision :

- garder SimLingo comme conducteur principal;
- ajouter un Dreamer externe comme complement;
- utiliser des traces closed-loop pour entrainer/evaluer ce complement.

## 11. Dreamer PPO v1

Le repo `youma2003/dreamer_ppo_carla` a ete analyse et integre. Le Dreamer PPO
v1 n'a pas ete utilise comme pilote autonome complet, mais comme adaptateur
autour de SimLingo.

Principe retenu :

- SimLingo produit l'action de base;
- le Dreamer observe l'etat et l'action SimLingo;
- il propose ou score des alternatives;
- le runtime choisit une action si elle semble meilleure.

Modes historiques crees pendant l'experimentation :

- shadow / visual only;
- guard strict;
- guard balanced;
- accident/overtake recovery;
- full candidate scorer.

Ces noms ont ensuite ete juges trop nombreux et trop confus. La direction finale
est de simplifier l'interface :

- `Off - native SimLingo`;
- `Dreamer PPO`;
- `Dreamer SDBS`;
- variantes RL no-guard separees.

## 12. Guards du Dreamer PPO v1

Le Dreamer v1 a eu besoin de protections runtime pour devenir stable.

Protections ajoutees progressivement :

- verification de la voie gauche avant depassement;
- verification du vehicule derriere a gauche;
- verification des vehicules en face sur la voie opposee;
- blocage si le depassement est dangereux;
- commit de depassement quand un gap est suffisamment clair;
- retour sur la voie apres obstacle;
- prevention d'offroad;
- prevention de collision laterale.

Question critique : est-ce de la triche ?

Reponse retenue :

- oui, si on pretend que c'est une policy pure end-to-end;
- non, si on le presente comme un systeme hybride learned world-model + runtime
  safety logic autour de SimLingo;
- il faut donc etre transparent dans les rapports et KPI.

Conclusion : le Dreamer PPO garde est une contribution viable, mais ce n'est pas
un Dreamer pur. C'est un complement learned/guarded de SimLingo.

## 13. Resultat qualitatif du Dreamer PPO v1

Le Dreamer PPO v1 a ete le premier module a donner un resultat visuellement
convaincant :

- contournement d'accident sur Town10HD;
- reprise de route apres depassement;
- reduction des blocages ou SimLingo natif restait immobile;
- meilleure gestion de certains obstacles.

Limites observees :

- parfois trop prudent, bloquant longtemps;
- parfois trop agressif si les seuils de gap etaient mal regles;
- possible offroad ou collision si les checks lateraux/opposes etaient
  insuffisants;
- forte sensibilite aux changements de code/checkpoint.

Decision :

- garder Dreamer PPO v1 comme meilleur module pratique actuel;
- ne pas le confondre avec une policy RL pure;
- le comparer a SimLingo natif via KPI et runs visuels.

## 14. Dreamer SDBS

Le repo `youma2003/dreamer_ppo_carla` a ensuite ete mis a jour avec une variante
SDBS.

But du SDBS :

- ajouter une strategie de search/diverse beam;
- ameliorer la selection d'actions;
- comparer PPO baseline, Dreamer PPO et SDBS.

Ce qui a ete fait :

- clonage du repo mis a jour;
- mise de cote des anciennes traces SDBS dans `trash`;
- integration comme option dashboard;
- creation d'un mode accident/overtake similaire au v1;
- collecte de donnees avec SimLingo + Dreamer v1 comme teacher;
- entrainement offline;
- tests closed-loop;
- creation de zips/export pour transmettre a la personne responsable du repo.

Constat critique :

- SDBS n'a pas atteint la robustesse du Dreamer PPO v1;
- il a parfois contourne mais provoque collision/offroad;
- il s'est parfois bloque;
- les donnees de collecte contenaient possiblement des comportements imparfaits;
- il faut encore retravailler le protocole d'entrainement ou demander correction
  upstream.

Decision :

- garder SDBS comme piste de comparaison;
- ne pas le presenter comme meilleur que v1 tant que ses KPI ne le prouvent pas;
- exporter les checkpoints/logs pour que la responsable du repo puisse analyser.

## 15. Collecte de donnees Dreamer

Un mode `Collect Data` a ete ajoute pour generer des traces Action Dreaming /
Dreamer.

Objectif :

- collecter des episodes ou SimLingo + Dreamer v1 reussit a depasser proprement;
- transformer ces traces en dataset;
- entrainer SDBS ou des variantes complementaires.

Problemes rencontres :

- une collecte peut contenir un accident ou un depassement dangereux;
- une mauvaise collecte peut degrader le modele;
- il faut filtrer les traces avant training;
- collecter uniquement "ce qui s'est passe" n'est pas suffisant : il faut savoir
  si le comportement etait bon.

Decision :

- supprimer ou ignorer les dernieres collectes rattees;
- auditer les datasets avant training;
- ne pas entrainer sur une collecte si le teacher a cause accident/offroad.

## 16. CoT externe

On a explore l'idee d'une CoT externe qui observe l'image camera et raisonne en
langage naturel.

Clarification importante :

- la CoT ne doit pas prendre le volant;
- elle doit servir d'observateur/explicateur;
- elle peut fournir un diagnostic : accident, pedestrian, vehicle ahead, red
  light, blocked lane;
- elle peut eventuellement eclairer le Dreamer ou l'analyse KPI plus tard.

Implementation :

- integration Qwen2-VL local en mode CoT;
- affichage dans la fenetre Pygame;
- modes mock et Qwen local;
- intervalle configurable.

Limites observees :

- la CoT peut manquer un accident visible;
- elle peut dire "no immediate hazard" alors qu'un obstacle est present;
- elle est lente et dependante du cadrage camera;
- elle n'est pas fiable comme controleur.

Decision :

- garder la CoT comme panneau d'observation/explicabilite;
- ne pas lui donner le controle;
- ne pas compter sur elle comme preuve de securite tant qu'elle n'est pas
  evaluee systematiquement.

## 17. Parenthese Axis 1 / CoT dangereux

Un travail externe a fourni :

- `cot_dataset.jsonl`;
- un rapport PDF;
- un repo Simple-carla-WAM;
- un entrainement Dreamer-PPO/CoT sur donnees dangereuses.

Ce qui a ete fait :

- preparation d'une VM GPU;
- installation dependances;
- correction notebook Colab vers chemins locaux;
- gestion erreurs PyTorch/CUDA;
- automatisation de recuperation des resultats;
- archivage du dossier Axis 1 training.

Decision critique :

- c'est une piste interessante;
- ce n'est pas encore integre comme composant efficace du pipeline principal;
- on ne le melange pas aux resultats SimLingo + Dreamer tant qu'il n'est pas
  teste closed-loop dans notre dashboard.

## 18. Parenthese repo Maram / Dreamer

Un autre repo Dreamer a ete teste a part.

Problemes rencontres :

- environnement Python/CARLA contraignant;
- entrainement local/VM instable;
- erreurs de dependances;
- resultat pas directement integrable dans SimLingo;
- risque de refaire un agent autonome CARLA generique plutot qu'un complement
  de SimLingo.

Decision :

- garder comme parenthese technique;
- ne pas l'integrer au pipeline principal;
- revenir au couple SimLingo + Dreamer complementaire.

## 19. SUMO mirror

Un bridge CARLA/SUMO a ete cree pour afficher une simulation CARLA en parallele
dans SUMO GUI.

Objectif :

- voir la simulation 3D dans CARLA/Pygame;
- voir la projection 2D dans SUMO;
- preparer des attaques trafic;
- garder route/scenario Bench2Drive comme base.

Ce qui a marche :

- affichage simultane CARLA + SUMO;
- synchronisation visuelle de certaines entites;
- lancement depuis dashboard;
- test sur Town04 puis routes Bench2Drive.

Limites :

- les XML Bench2Drive ne sont pas des fichiers SUMO directement utilisables;
- SUMO et CARLA ont des formats differents;
- la synchronisation parfaite des vehicules/feux demande un vrai pont d'etat,
  pas juste deux simulations paralleles;
- il a fallu clarifier si SUMO controlait CARLA ou si SUMO etait miroir.

Decision :

- garder SUMO mirror comme base;
- faire en sorte que les attaques modifient les vrais objets CARLA si elles
  doivent impacter SimLingo;
- ne pas se contenter d'un effet visuel SUMO.

## 20. Attaques trafic et TwinSentinel

Le projet TwinSentinel a ete ajoute pour fournir une webapp d'attaques SUMO.

Objectif :

- lancer SimLingo/CARLA/SUMO depuis notre interface;
- ouvrir la console TwinSentinel;
- choisir une attaque;
- appliquer l'attaque sur la simulation en cours;
- voir l'effet dans SUMO et CARLA.

Probleme rencontre :

- une ancienne interface `SUMO_Project` a ete melangee par erreur;
- elle ne correspondait pas au repo TwinSentinel demande;
- il fallait repartir du repo `E-Mehdi-Boulharts/TwinSentinel_Project`.

Correction :

- clonage complet avec Git LFS;
- integration TwinSentinel dans `experiments/TwinSentinel_Project`;
- script `run_twinsentinel_attack_console.sh`;
- endpoint dashboard pour ouvrir la console d'attaques;
- nettoyage progressif des anciennes attaques non desirees.

Etat critique :

- les attaques traffic-light doivent modifier CARLA pour etre scientifiquement
  valables;
- si seul SUMO change, SimLingo ne sera pas vraiment perturbe;
- l'objectif final reste : meme route, meme scenario, environnement attaque.

## 21. Traffic-light attack

Un mode d'attaque "all red" a ete implemente/teste.

Principe :

- warmup;
- passage des feux CARLA au rouge;
- duree d'attaque configurable;
- observation dans Pygame;
- possible penalisation Bench2Drive si SimLingo grille le feu.

Decision :

- l'attaque doit etre reelle dans CARLA;
- elle doit etre visible dans Pygame;
- elle doit aussi etre visible dans SUMO quand le miroir est actif.

## 22. Overlay feux CARLA

Un overlay de feux a ete ajoute.

Objectif :

- afficher l'etat des feux dans la vue Pygame;
- ne pas se limiter au feu devant l'ego;
- aider a verifier si l'attaque modifie vraiment les feux CARLA.

Etat :

- option dans dashboard;
- distance/range configurable;
- affichage de badges.

## 23. SAFE-DREAM KPI

Le papier et le code SAFE-DREAM KPI ont ete integres pour comparer les modeles.

Objectif :

Comparer les memes familles de metriques pour :

- SimLingo natif;
- SimLingo + Dreamer PPO;
- SimLingo + Dreamer SDBS;
- variantes RL no-guard.

Types de KPI :

- route completion;
- driving score;
- collisions;
- offroad / outside route;
- red light;
- blocked agent;
- override rate;
- proxy SAFE-DREAM DQI;
- metriques issues des traces Dreamer.

Problemes rencontres :

- chevauchement visuel des textes;
- certaines colonnes sans runs;
- DQI parfois identique entre deux Dreamers parce que les traces/proxies etaient
  similaires ou insuffisantes;
- distinction entre "scored" et "started" pas toujours claire.

Corrections :

- meilleure lisibilite des cartes KPI;
- separation des colonnes par mode;
- lecture de tous les runs observes;
- indication claire si un run Bench2Drive n'a pas produit de score eligible.

Critique :

- les KPI SAFE-DREAM sont utiles pour comparer;
- certains restent des proxies tant qu'on n'a pas une evaluation complete;
- il faut eviter de revendiquer un gain avec seulement une impression visuelle.

## 24. GitHub et reproductibilite

Le projet a ete prepare pour GitHub prive.

Actions :

- creation README;
- docs install fresh machine;
- requirements;
- environment Conda;
- lock/freeze files;
- `.gitignore`;
- Git LFS pour checkpoints;
- explication de ce qui n'est pas pousse : CARLA, videos, logs, datasets,
  caches, tokens, backups, poids Hugging Face lourds.

Clarification importante :

- personne n'a acces au PC local via GitHub;
- les tokens Hugging Face ne sont pas inclus;
- les donnees privees ne sont pas incluses;
- une personne externe doit suivre la doc pour telecharger CARLA, SimLingo et
  les dependances.

## 25. Probleme de dossiers vla-av / dev / stable

Pendant le projet, plusieurs dossiers ont existe :

- `vla-av`;
- `vla-av-simlingo-dev`;
- `vla-av-simlingo-stable-...`.

Cela a cree de la confusion :

- certains scripts fonctionnaient dans un dossier et pas l'autre;
- des changements ont pu etre faits dans le mauvais arbre;
- le user a observe des regressions et incoherences.

Decision finale :

- le projet actif est `~/Desktop/vla-av`;
- on ne continue pas dans le dossier dev;
- la version stable sert seulement de backup/reference;
- toute nouvelle modification doit se faire dans `vla-av`.

Critique : cette confusion a probablement explique une partie des regressions
sur Dreamer v1/SDBS. La mise sous Git doit maintenant eviter de revivre ce type
de perte de controle.

## 26. Nettoyage et archivage

Plusieurs elements ont ete deplaces ou mis de cote :

- anciennes variantes SDBS non concluantes;
- anciens modes experimentaux;
- collectes ratees;
- dossiers inutiles ou non relies au pipeline;
- exports pour collegues.

Principe retenu :

- ne pas supprimer brutalement;
- archiver dans `trash`, `backups` ou `exports`;
- garder le projet actif le plus lisible possible.

## 27. RL no-guard : clarification

Une demande forte a ete formulee : avoir du vrai RL online sans guards.

Clarification importante :

- les modes `Dreamer PPO` et `Dreamer SDBS` classiques restent guarded;
- les modes `Dreamer PPO RL no-guard` et `Dreamer SDBS RL no-guard` doivent
  etre sans guard au volant;
- pas de collision shield;
- pas de recovery;
- pas de gap rule;
- pas de veto TTC;
- pas de safety override.

Ils restent cependant des complements de SimLingo :

- SimLingo produit l'action de base;
- le Dreamer RL produit une action residuale/remplacement;
- l'apprentissage se fait via reward/malus.

## 28. Difference entre demo et training online

Un probleme important est apparu : si chaque simulation web entraine
automatiquement, un mauvais run peut empoisonner le checkpoint.

Exemple observe :

- un run mauvais a mis a jour un checkpoint;
- le modele a ensuite appris a freiner/bloquer;
- il a fallu restaurer un checkpoint depuis backup.

Decision recente :

- par defaut, une simulation web est en mode `run only`;
- aucun checkpoint n'est mis a jour;
- l'utilisateur doit choisir explicitement `RL update: Training session`;
- un stop manuel n'entraine pas;
- un rollout catastrophique ne promeut pas le checkpoint.

Important :

- ce filtrage n'est pas un guard de conduite;
- c'est une protection de training/checkpoint;
- le mode RL no-guard reste sans guard pendant le run.

## 29. Reward/malus RL online

Le reward RL online doit encourager :

- progression route;
- depassement propre;
- retour dans la voie;
- absence de collision;
- absence d'offroad;
- respect feux/stop;
- non-blocage;
- conduite fluide.

Il doit penaliser :

- collision vehicule;
- collision pieton/VRU;
- offroad;
- red light;
- stop violation;
- blocked agent;
- depassement trop proche d'un vehicule a gauche;
- depassement avec vehicule en face trop proche;
- retour dangereux;
- conduite saccadee;
- freinage inutile prolonge.

Critique :

- ce n'est pas une solution magique;
- le reward shaping est difficile;
- trop de malus peut rendre le Dreamer trop prudent;
- pas assez de malus peut le rendre dangereux.

## 30. Dernier etat RL online

Les modes no-guard sont maintenant separes des modes gardes.

L'interface contient :

- `Dreamer PPO`;
- `Dreamer SDBS`;
- `Dreamer PPO RL no-guard`;
- `Dreamer SDBS RL no-guard`;
- `RL update: Off - run only`;
- `RL update: Training session`.

Dernier run observe :

- route `bench2drive_148`;
- Town10HD;
- scenario Accident;
- seed `675550`;
- mode `dreamer_ppo_rl_noguard`;
- training desactive;
- 100% route completion;
- 0 collision;
- 0 offroad;
- 0 red light;
- 0 blocked agent;
- echec uniquement sur `MinSpeedTest`.

Interpretation :

- tres bon run qualitatif;
- bon candidat a relancer en `Training session`;
- pas encore appris car `training=0`.

## 31. Ce qui est actif aujourd'hui

Composants actifs :

- SimLingo closed-loop natif;
- dashboard web local;
- Pygame POV HD/Epic;
- overlays waypoints;
- replay video accelere;
- route/scenario selector;
- Dreamer PPO guarded;
- Dreamer SDBS guarded;
- Dreamer PPO RL no-guard;
- Dreamer SDBS RL no-guard;
- CoT externe Qwen panel;
- SUMO mirror;
- TwinSentinel attack console;
- traffic-light overlay;
- SAFE-DREAM KPI dashboard;
- GitHub/private repo preparation.

## 32. Ce qui est mis de cote

Mis de cote ou non revendique comme final :

- routes longues custom comme benchmark principal;
- safety mode / fast driving / turn guard comme evaluation SimLingo;
- Dreamer pur autonome remplaçant SimLingo;
- anciennes variantes multiples de Dreamer dans l'interface;
- SDBS comme meilleur modele tant qu'il n'a pas de KPI solide;
- CoT comme controleur;
- Axis 1 CoT training comme composant actif;
- repo Maram Dreamer comme composant actif;
- anciennes interfaces SUMO_Project hors TwinSentinel;
- collectes Dreamer contenant accident/offroad.

## 33. Position scientifique actuelle

La contribution la plus solide n'est pas "SimLingo remplace par un autre modele".

La contribution defendable est :

> SimLingo reste le VLA de base. Nous construisons autour de lui une plateforme
> CARLA/Bench2Drive/SUMO permettant de visualiser, attaquer, comparer et
> ameliorer son comportement. Le Dreamer PPO guarded est une premiere extension
> efficace pour certains cas d'accident/overtake. Les variantes SDBS et RL
> no-guard sont des pistes experimentales separees, a valider par KPI.

## 34. Risques restants

Risques techniques :

- lenteur extreme de SimLingo natif;
- dependance forte a CARLA/CUDA/env local;
- Hugging Face remote code warnings;
- confusion possible entre checkpoint garde et no-guard;
- datasets/collectes de qualite variable;
- RL online fragile si le reward est mal calibre;
- TwinSentinel/SUMO/CARLA sync encore delicate.

Risques scientifiques :

- confondre guard code avec apprentissage pur;
- presenter un succes visuel sans KPI;
- comparer des runs non identiques;
- utiliser des seeds/routes differentes;
- entrainer sur des rollouts mauvais;
- declarer SDBS meilleur sans preuve.

## 35. Prochaine logique propre

La prochaine etape la plus saine :

1. Geler le projet actif dans Git.
2. Garder les modes guarded stables intacts.
3. Lancer des runs comparatifs memes routes/seeds :
   - SimLingo natif;
   - Dreamer PPO;
   - Dreamer SDBS;
   - Dreamer PPO RL no-guard;
   - Dreamer SDBS RL no-guard.
4. Pour RL no-guard, utiliser `Training session` seulement sur des scenarios
   choisis.
5. Ne promouvoir un checkpoint que si :
   - pas de collision;
   - pas d'offroad;
   - route completion bonne;
   - comportement visuel acceptable;
   - KPI meilleurs ou au moins non degrades.
6. Ensuite seulement, relancer SUMO/TwinSentinel attacks pour tester robustesse.

## 36. Conclusion critique

Depuis l'integration de SimLingo, le projet est passe d'un simple lancement
closed-loop CARLA a une plateforme complete :

- demonstration visuelle;
- replay video;
- selection Bench2Drive;
- comparaison KPI;
- Dreamer complement;
- CoT explicative;
- SUMO mirror;
- attaques TwinSentinel;
- preparation RL online.

Le point le plus important a retenir est la separation des roles :

- SimLingo natif = baseline.
- Dreamer PPO guarded = amelioration pratique/hybride.
- Dreamer SDBS = piste experimentale.
- RL no-guard = recherche en cours, sans guard au volant, mais avec controle de
  promotion checkpoint.
- CoT = observation/explication, pas controle.
- SUMO/TwinSentinel = red-team evaluation.

Cette separation evite de melanger demonstration, entrainement, evaluation et
recherche. Elle rend le projet plus defendable et permet de mesurer proprement
si une extension outperforme vraiment SimLingo.
