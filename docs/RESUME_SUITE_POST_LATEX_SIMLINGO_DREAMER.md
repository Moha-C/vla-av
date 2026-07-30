# Resume de la suite du projet apres le document LaTeX existant

Ce resume commence apres la derniere etape deja presente dans le document LaTeX fourni, c'est-a-dire apres l'integration initiale de SimLingo, du dashboard web, des routes Bench2Drive et des premiers modes de visualisation. Il ne reprend pas les parentheses hors pipeline final, notamment les trainings CoT/Axis 1 sur VM et les essais du repo Maram, car ils ne sont pas encore integres comme composant efficace dans la pipeline principale CARLA-SimLingo.

## 1. Stabilisation de la baseline SimLingo

La priorite a ete de stabiliser SimLingo comme baseline scientifique fiable. Les modes artificiels ajoutes pour accelerer ou "securiser" la conduite ont ete remis en question, puis ecartes pour conserver une version native de SimLingo lorsque l'objectif est de mesurer le modele. La logique retenue est claire : SimLingo doit rester le conducteur principal, sans autopilot cache ni correction externe quand on evalue la baseline.

La visualisation Pygame a ete amelioree pour devenir exploitable en demonstration : qualite Epic/1080p, choix du point de vue, rendu chase/wheel/front/top, reduction des frames noires, overlay des trajectoires et enregistrement video. L'overlay reprend l'esprit de la demo SimLingo : rouge pour les waypoints de trajectoire predits, vert pour les waypoints lies a la vitesse, bleu pour les target points de route. Comme CARLA et SimLingo peuvent tourner lentement en closed-loop natif, une logique de replay accelere a ete ajoutee afin de revoir une simulation en x5, x50 ou plus sans modifier la conduite pendant le run.

Une separation entre version stable et version de developpement a ete mise en place. La version stable sert de reference intacte lorsque SimLingo fonctionne correctement. La version de developpement sert aux ajouts experimentaux : SUMO, Dreamer Guard, KPI et nouveaux modes de comparaison.

## 2. Routes longues et limites observees

Un generateur de routes longues Bench2Drive custom a ete essaye pour parcourir de grandes parties de map. La premiere tentative a echoue parce que le fichier XML genere ne contenait pas de scenario compatible avec le leaderboard, ce qui provoquait une erreur `scenario_configs[0]`. Une version avec scenario neutre a ensuite permis de lancer la route.

Cependant, les routes tres longues creees par collage de keypoints Bench2Drive ont montre une limite pratique : SimLingo pouvait rouler, mais la conduite devenait instable sur certains virages ou raccords, avec zigzags, offroad ou perte de controle. Conclusion d'ingenierie : pour les tests scientifiques et les comparaisons, il vaut mieux rester sur les routes/scenarios natifs Bench2Drive plutot que d'utiliser des routes collees artificiellement. Les routes custom restent utiles pour la demo visuelle, mais pas comme benchmark principal.

## 3. Integration SUMO / CARLA mirror

Le projet SUMO externe a ete analyse pour migrer l'idee de red-team traffic-light vers CARLA. La solution retenue n'est pas un simple effet visuel : l'attaque doit modifier les vrais objets `traffic_light` de CARLA. Ainsi, les feux deviennent rouges dans Pygame, les cameras de SimLingo voient ces feux rouges, et Bench2Drive peut penaliser le comportement si le modele les ignore.

Un mode SUMO mirror a ete mis en place. Il permet de lancer une simulation CARLA/SimLingo avec la vue Pygame en meme temps qu'une fenetre SUMO GUI 2D. Le test sur Town04 a valide le principe : CARLA affiche la scene 3D pendant que SUMO montre une carte 2D synchronisee. Le dashboard web permet maintenant de lancer le mode `CARLA POV + SUMO mirror`, ce qui donne une base solide pour les futures attaques traffic-light et traffic-flow.

La philosophie retenue est : meme route, meme scenario Bench2Drive, mais environnement dynamique perturbe. Cela permet de mesurer si SimLingo respecte, ignore, bloque ou echoue sous attaque d'infrastructure.

## 4. Dreamer-PPO v1 autour de SimLingo

Le repo `youma2003/dreamer_ppo_carla` a ete analyse puis adapte non pas comme remplacement direct de SimLingo, mais comme module de prediction/selection d'action autour de SimLingo. L'idee finale est un `Dreamer Guard` : SimLingo propose l'action principale, puis un monde modele court horizon score plusieurs actions candidates autour de cette action. Le Dreamer n'a le droit d'intervenir que lorsqu'il predit une reduction claire du risque avec une degradation minimale de la progression.

Le Dreamer Guard v1 a ete valide offline sur un dataset Action Dreaming filtre Town12/Town13 :

- 2528 samples scores.
- World model entraine sur 2509 transitions.
- Meilleur checkpoint : `best_world_model.pt`.
- Validation loss du world model : environ 0.937.
- State MAE normalisee : environ 0.069.
- Risk MAE : environ 0.123.
- Mode guard strict : environ 4.07 % d'overrides.
- Mode guard plus permissif : environ 5.42 % d'overrides.

Le test important est que le Dreamer pur, utilise comme remplacement complet de SimLingo, n'est pas assez fiable : il choisit souvent des actions qui ameliorent son score appris mais degradent les metadonnees originales. En revanche, le mode Guard est viable, car il intervient rarement et seulement sur des cas ou le risque baisse.

En closed-loop, le mode accident/overtake guard a apporte une amelioration visible : sur un scenario ou SimLingo natif restait bloque derriere un accident, le Dreamer Guard a permis de proposer un contournement. Cette amelioration doit etre presentee comme un adaptateur de securite learned world-model autour du VLA, pas comme une modification du backbone SimLingo.

## 5. Dreamer-PPO v2 / SDBS du repo youma2003

Le fichier `RUNME.md` distant du repo `youma2003/dreamer_ppo_carla` precise trois variantes a comparer :

- `PPO baseline` : reference sans world model ni dreaming.
- `Dreamer-PPO` : world model + one-step dreaming.
- `SDBS Dreamer-PPO` : version complete avec diverse beam search et curriculum.

Le runbook impose des KPI orientes securite :

- VRU safety en priorite : collisions avec pietons/cyclistes, near-misses, min TTC, distance moyenne aux VRU.
- Vehicle safety : collisions vehicules, near-misses, rear incidents.
- Performance : mean return, route completion, success rate.
- Scores headline : `vru_safety_score` et `composite_score`, avec classement prioritaire par securite VRU.

La version v2/SDBS est donc traitee comme une piste prometteuse, mais elle ne doit pas etre declaree meilleure tant que les CSV `logs/baseline.csv`, `logs/dreamer.csv` et `logs/sdbs.csv` n'ont pas ete produits et compares. Des essais qualitatifs ont montre une limite : en depassement sur route a double sens, le Dreamer peut contourner l'obstacle mais prendre trop de risque face aux vehicules venant en sens inverse. Il faut donc le garder en mode shadow/guard et ajouter des contraintes de voie opposee avant tout usage comme controller plus libre.

## 6. Dashboard comparatif

Le dashboard web a ete etendu avec une fenetre de comparaison KPI entre :

- SimLingo natif.
- Dreamer Guard v1.
- Dreamer v2 / SDBS du repo youma2003.

La fenetre lit automatiquement les resultats disponibles localement :

- Resultats Bench2Drive JSON pour la baseline SimLingo.
- `summary.json` offline du Dreamer Guard v1.
- CSV v2/SDBS si les logs du runbook existent.

Quand un resultat n'existe pas encore, l'interface l'indique explicitement comme `pending` au lieu d'inventer une valeur. Cela garde la demonstration honnete et defendable. La comparaison reprend les KPI du `RUNME.md`, notamment la priorite VRU safety et le `composite_score`.

## 7. Position actuelle et prochaine etape logique

L'etat actuel du projet est le suivant :

- SimLingo natif est la baseline closed-loop fonctionnelle.
- La visualisation est exploitable pour demo : Pygame HD/Epic, overlays, replay accelere.
- SUMO mirror fonctionne et prepare les futures attaques dynamiques.
- Dreamer Guard v1 est la contribution la plus solide a ce stade : il ne remplace pas SimLingo, mais ajoute un world-model safety guard.
- Dreamer v2/SDBS est integre conceptuellement dans la comparaison, mais doit encore produire des KPI complets avant d'etre revendique comme amelioration.

La suite logique est de lancer une campagne comparative stricte :

1. Meme route et meme seed en SimLingo natif.
2. Meme route et meme seed avec Dreamer Guard v1.
3. Meme route et meme seed avec Dreamer v2/SDBS lorsque ses logs/checkpoints sont disponibles.
4. Comparaison sur accidents, VRU, cut-in, traffic-light et scenarios SUMO attacks.
5. Analyse finale : le gain n'est accepte que s'il reduit collisions/near-misses/blocages sans degrader fortement route completion ni introduire de comportements dangereux en sens inverse.

Conclusion critique : la direction viable pour outperform SimLingo n'est pas de remplacer brutalement le VLA, mais d'ajouter un module learned world-model qui agit comme guard explicable, mesurable et limite. C'est ce qui permet de presenter une contribution originale sans perdre la robustesse de la baseline SimLingo.
