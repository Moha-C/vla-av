Tu es Claude et tu dois rédiger une documentation claire, professionnelle et structurée pour mon projet de stage VLA-AV/CARLA. Tu dois reprendre l'historique depuis le 12 mai 2026 vers 10h12 et intégrer toutes les évolutions importantes ci-dessous. Écris en français, avec une structure de rapport technique: contexte, objectifs, pipeline, données, entraînements, résultats, problèmes rencontrés, décisions techniques, état actuel, prochaines étapes, commandes utiles.

Contexte général:
- Projet: VLA-AV, intégration et test de modèles Vision-Language-Action pour conduite autonome dans CARLA.
- Objectif: obtenir un agent closed-loop capable de conduire en CARLA, respecter la signalisation, gérer feux rouges/verts, stops, véhicules, piétons, cyclistes, scooters/motos/VRU, et produire des résultats exploitables pour mon stage.
- Modèles étudiés: NVIDIA Alpamayo-R1-10B et SimLingo de RenzKa.

Dataset CARLA/Alpamayo:
- Dataset final principal: `data/alpamayo_carla_dataset_b2008_base_combined/manifest.jsonl`.
- Taille validée: environ 171,905 frames, 1,333 clips/runs réels d'environ 129 frames chacun.
- Contrôles qualité: aucune image manquante, `ego_history_xyz` de longueur 16, `ego_future_xyz` de longueur 64, chaînes de causalité présentes.
- Tags principaux observés:
  - `expert_accelerating`: ~99,849
  - `expert_braking`: ~59,211
  - `expert_stopped`: ~43,351
  - `vehicles_visible`: ~55,312
  - `vru_visible`: ~64,173
  - `vru_rider_visible`: ~51,003
  - `vru_motorcycle_visible`: ~36,362
  - `vru_in_ego_corridor`: ~35,867
  - `vru_near_path`: ~35,143
  - `vehicle_in_ego_corridor`: ~18,704
  - `traffic_light_red`: ~35,218
  - `traffic_light_green`: ~4,170
  - `stop_sign_near`: ~2,834
  - `stop_sign_close`: ~2,422
- Les manifestes ont été enrichis avec `semantic_context`, `driving_policy_tags`, `rule_context`, `situational_instruction`, `chain_of_causation`.
- Insistance particulière sur les traces de raisonnement / chain of causation: perception evidence, rule evaluation, expert action rationale, VRU priority, signalisation, traffic context.
- Le recalcul complet de sémantique depuis segmentation était trop lent sur CPU; une version filtrée VRU a été utilisée pour rendre `vru_near_path` / `vru_in_ego_corridor` beaucoup plus utiles.

Conversion Alpamayo SFT:
- Dataset converti vers `data/alpamayo_sft_carla_b2008_base`.
- Audit conversion:
  - samples: 171,905
  - train: 168,422
  - val: 3,483
  - fatal errors: 0
  - missing images/history/future/assistant text/chain of causation: 0.
- Fichier d'audit: `artifacts/full_sft_conversion_audit.json`.
- Fichiers SFT: `train.jsonl`, `val.jsonl`, `summary.json`, `CONVERSION_OK.json`.
- Un bridge dataset officiel a été ajouté pour Alpamayo: `external/alpamayo_official/finetune/sft/datasets/carla_jsonl_dataset.py`.

Environnement et entraînement Alpamayo:
- Environnement officiel Alpamayo: `external/alpamayo_official/ar1_venv`.
- Installation CUDA toolkit 12.8 conda séparée pour compiler `flash-attn`.
- `flash-attn==2.8.3` installé dans l'environnement officiel Alpamayo.
- Vérification GPU VM: 8x NVIDIA B200.
- Full SFT officiel lancé en deux stages:
  - Stage 1: VLM SFT, config `sft_carla_stage1`, 2 epochs, 10,528 steps, environ 15h, loss finale ~0.336, train_loss ~0.579.
  - Stage 2: trajectory diffusion expert / expert head, relancé après crash post-stage1, puis terminé.
- Des crashes `SIGSEGV` sont apparus en fin de stage ou autour de transitions, mais les checkpoints ont été sauvegardés.
- Artifacts récupérés localement:
  - Archive officielle full SFT: `vm_backups/official_sft/official_alpamayo_full_sft_20260515_180937.tar.zst`, environ 23G.
  - Checkpoint stage2 final: `vm_backups/official_sft/intermediate/stage2/checkpoint-10528`, environ 30G.
- Des watchers/rsync locaux avaient été mis en place pour récupérer checkpoints et artifacts depuis la VM.

Résultats Alpamayo:
- Tests closed-loop locaux avec Alpamayo fine-tuné insatisfaisants:
  - off-road,
  - non-respect apparent de feux/stops,
  - comportement trop rectiligne,
  - dépendance forte à lane assist,
  - confusion dans l'UI autour de `Control: VLA`.
- Diagnostic technique: le problème vient en grande partie de l'intégration closed-loop. Alpamayo a été branché comme s'il était un agent CARLA complet, mais sans route planner/target point robuste. Sans objectif de navigation haut niveau, un VLA peut prédire une trajectoire locale mais ne peut pas savoir quelle branche choisir à une intersection.
- Décision: garder Alpamayo, mais tester SimLingo comme agent CARLA closed-loop officiel mieux adapté à Bench2Drive/Leaderboard.

SimLingo:
- Sources:
  - LearnOpenCV SimLingo tutorial.
  - GitHub: `https://github.com/RenzKa/simlingo`.
  - Modèle Hugging Face: `RenzKa/simlingo`.
- SimLingo est un VLA closed-loop pour CARLA/Bench2Drive:
  - Entrées: caméra RGB, speed/GPS/compass, route/target point/command.
  - Backbone: InternVL2-1B.
  - Architecture: VLM + langage + driving adaptor.
  - Sorties: predicted route waypoints + speed waypoints.
  - Contrôle final: PID vers steer/throttle/brake.
- Structure importante:
  - Agent: `external/simlingo/team_code/agent_simlingo.py`
  - Modèle: `external/simlingo/simlingo_training/models/driving.py`
  - Adaptors/heads: `external/simlingo/simlingo_training/models/adaptors/adaptors.py`
  - Config checkpoint: `models/simlingo_hf/simlingo/.hydra/config.yaml`
  - Checkpoint: `models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- Environnement séparé: conda env `simlingo`, Python 3.8, torch 2.2.0+cu121, CARLA 0.9.15.
- Scripts ajoutés:
  - `scripts/setup_simlingo_env.sh`
  - `scripts/download_simlingo_model.sh`
  - `scripts/run_simlingo_local_eval.sh`
  - `scripts/run_simlingo_with_pov.sh`
  - `scripts/carla_ego_viewer.py`
  - `scripts/simlingo_dashboard.py`
  - `scripts/run_simlingo_dashboard.sh`
  - `scripts/stop_simlingo_dashboard.sh`
  - `scripts/list_simlingo_routes.py`
  - `scripts/install_carla_additional_maps_0915.sh`

Maps CARLA:
- Au début, seules les maps de base étaient visibles: Town01-05, Town10HD.
- Les routes Bench2Drive Town12 échouaient avec `RuntimeError: Map 'Town12' not found`.
- Installation/import de maps additionnelles CARLA 0.9.15.
- Détection corrigée pour chercher récursivement les `.umap`, car les maps additionnelles sont dans des sous-dossiers.
- État actuel:
  - Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD, Town11, Town12, Town13, Town15 sont détectées.
  - Les 220 routes Bench2Drive sont maintenant compatibles.

Dashboard SimLingo:
- Interface web localhost ajoutée pour lancer SimLingo, puis simplifiée:
  - choix map,
  - scénario: Any / VRU / traffic light / stop / junction,
  - route random ou précise,
  - seed,
  - qualité CARLA,
  - POV: chase / wheel / front / top,
  - météo visuelle: Day / Soft clouds / Sunset / route weather,
  - bouton GO stylé avec DA inspirée de l'interface de Mehdi.
- Décision récente: le dashboard web ne doit plus afficher de vidéo ni de live log terminal. Il sert uniquement de panneau de paramètres. Quand l'utilisateur clique sur GO, le dashboard lance `scripts/run_simlingo_with_pov.sh`, qui ouvre une fenêtre Pygame séparée et laisse les logs sortir dans le terminal normal.
- Palette utilisée:
  - background `#f4efe6`
  - panel `#fffaf0`
  - ink `#1f2937`
  - accent `#d97706`
  - accent-2 `#0f766e`
  - danger `#b91c1c`
  - grid `#e5dccd`
- Bug corrigé: `Address already in use` sur port 8765; le script choisit automatiquement un port libre suivant.
- Bug corrigé: le script lançait CARLA puis Bench2Drive relançait son propre CARLA offscreen; le double lancement a été supprimé.
- Bug corrigé: détection `route_town` avec `cut`.
- Bug corrigé/amélioration: l'ancien stream web CARLA a été supprimé car il pouvait créer/détruire des acteurs caméra et provoquer des erreurs `failed to destroy actor` puis `Aborted (core dumped)`.
- Ajout d'un mode `wheel` pour une caméra plus proche du pare-brise/cockpit.
- `scripts/run_simlingo_with_pov.sh` a été corrigé en dernier lieu pour ouvrir la fenêtre Pygame en premier, puis lancer l'évaluation Bench2Drive/SimLingo. Le viewer affiche alors un écran d'attente et s'attache automatiquement à l'ego dès que CARLA et le scénario ont spawné la voiture. Cela évite le symptôme où SimLingo roulait en terminal mais aucune fenêtre Pygame n'apparaissait.
- Bug SimLingo corrigé ensuite: avec `SIMLINGO_FAST_DRIVING=1`, la branche sans génération langage plantait dans `DrivingModel.forward` avec `TypeError: tuple indices must be integers or slices, not tuple`, car `forward_model()` renvoie `(features, logits)` et l'ancien code passait le tuple entier à `split_outputs_by_adaptor`. Patch: `features, _ = self.forward_model(...)`.
- Nettoyage agent corrigé: `destroy()` ne suppose plus que `cfg.data_module.encoder` existe, ce qui évite un second crash OmegaConf pendant la fermeture après une erreur.
- Viewer Pygame corrigé: il tolère maintenant que l'ego disparaisse après fin/crash de route (`ego is None`) au lieu de lever `AttributeError: 'NoneType' object has no attribute 'is_alive'`.
- Viewer Pygame renforcé: il affiche explicitement `Connecting to CARLA`, `Waiting for SimLingo ego vehicle`, `Waiting for camera frames`; il ré-attache une nouvelle caméra si la map/scénario recrée l'ego ou détruit le capteur RGB. La sélection ego privilégie les rôles `hero`, `ego`, `ego_vehicle`, `hero0`, puis tombe sur un véhicule non-scenario/non-background.

Performance SimLingo:
- Problème observé: simulation très lente, ratio environ `0.062x`; la voiture n'était pas lente par vitesse, mais le simulateur avançait lentement car l'inférence SimLingo prenait trop de temps par tick.
- Cause probable: génération langage/CoT à chaque frame (`greedy_sample`, `max_new_tokens=100`) + absence de FlashAttention.
- Patch ajouté dans `external/simlingo/team_code/agent_simlingo.py`:
  - variable `SIMLINGO_FAST_DRIVING=1` par défaut,
  - désactive la génération langage par frame (`model.predict_language=False`),
  - désactive `use_cot`,
  - garde la prédiction de waypoints/action pour closed-loop driving.
- Le mode complet langage reste disponible avec `SIMLINGO_FAST_DRIVING=0`, mais il est beaucoup plus lent.

Commandes actuelles utiles:
- Arrêter dashboard/CARLA/SimLingo:
  `bash scripts/stop_simlingo_dashboard.sh`
- Lancer dashboard:
  `bash scripts/run_simlingo_dashboard.sh`
- Lister maps/routes:
  `python3 scripts/list_simlingo_routes.py`
- Lancer SimLingo avec Pygame POV chase rapide:
  `SIMLINGO_FAST_DRIVING=1 SIMLINGO_VISUAL_WEATHER=day SIMLINGO_VIEW_MODE=chase ROUTE_ID=145 SEED=$RANDOM PORT=2000 TM_PORT=8000 bash scripts/run_simlingo_with_pov.sh`
- Lancer une route VRU:
  `ROUTE_ID=145`, `146`, `147`, `151`, `175`, `202`, `203` ou random avec filtre dashboard.

Points à expliquer dans la documentation:
- Différence entre free-spawn et Bench2Drive route:
  SimLingo n'est pas un agent "lâché sans destination"; il suit une route/target point, comme un agent CARLA Leaderboard. Le scénario est closed-loop mais route et triggers sont définis par XML.
- Seed:
  même route + même seed donne un scénario très similaire; seed random varie le lancement.
- Pourquoi Alpamayo a échoué en closed-loop:
  absence de route planner/target point et mismatch pipeline/dataset.
- Pourquoi SimLingo est plus adapté:
  agent officiel Bench2Drive/Leaderboard, route planner, target points, PID, closed-loop.
- Limites:
  SimLingo reste CARLA/Bench2Drive-specific, pas véhicule réel.
  Les nombres de piétons/voitures/scooters dans le dashboard sont encore contrôlés par les routes/scénarios XML; un générateur custom sera nécessaire pour les choisir librement.
- Prochaine étape:
  ajouter overlay waypoints rouge/vert/bleu comme dans le tutoriel LearnOpenCV, améliorer cockpit/wheel POV, générer scénarios custom avec paramètres traffic/VRU, puis intégrer plus tard les red attacks SUMO de Mehdi.

Rédige la documentation finale comme un rapport clair, pas comme une conversation. Mets les commandes dans des blocs code, les résultats clés dans des tableaux, et termine par une section "État actuel et prochaines actions".
