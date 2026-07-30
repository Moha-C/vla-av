# Handoff Claude - reprise 100% locale du projet VLA-AV

Date de reprise locale: 2026-05-12.

Contexte utilisateur: le travail cloud/B200 est en pause, la suite doit se faire
entierement sur le PC local dans `~/Desktop/vla-av`. L'objectif prioritaire est
de repartir du pipeline fonctionnel actuel, nettoyer l'ancien projet, generer un
dataset CARLA -> Cosmos Transfer2.5 -> Alpamayo, puis produire un LoRA/adaptateur
rapide pret ce soir.

## Consignes utilisateur a respecter

1. Garder dans le projet principal uniquement le pipeline actuel et fonctionnel.
2. Mettre tout le legacy inutile dans `backup_1ere_version/`, sans suppression.
3. Reprendre de zero les images de fine-tuning et les videos generees, sauf les
   outputs NVIDIA Cosmos Transfer deja valides qui doivent rester accessibles.
4. Le dataset final d'entrainement doit utiliser les images photorealistes issues
   de Cosmos Transfer2.5, pas les frames CARLA brutes.
5. Les labels viennent de l'autopilot CARLA: steering, throttle, brake, ego state,
   historique/futur ego, et metadonnees code de la route.
6. Les prompts/instructions doivent couvrir: rester dans la voie, suivre les
   marquages, respecter feux rouges/verts, stops, fleches au sol, passages
   pietons, priorites, limites de vitesse, VRU, pietons, cyclistes, scooters,
   motos, vehicules, obstacles et reprise fluide quand la route est libre.
7. Le fine-tuning local doit etre rapide et pret ce soir. Si un vrai LoRA complet
   Alpamayo-1.5-10B est trop lourd/localement risque, preferer un adaptateur
   LoRA-like/residuel rapide au-dessus du planner Alpamayo.
8. Etape 3 plus tard: initialiser GitHub et expliquer quoi faire a chaque update.
9. Etape 4 plus tard: dockeriser de A a Z.

## Etat local important

Le repo courant est:

```text
/home/mohm/Desktop/vla-av
```

Apres verification utilisateur, le GPU local est visible:

```text
NVIDIA RTX A6000, 49140 MiB VRAM
Driver 580.142, CUDA 13.0
```

PyTorch dans `vla-av-step18` voit CUDA:

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 1
```

Verification initiale dans le shell Codex avant refresh:

```text
vla-av-step18: torch 2.7.1+cu128, cuda_available=False, cuda_count=0
cosmos venv:   torch 2.7.0+cu128, cuda_available=False, cuda_count=0
alpamayo venv: import alpamayo ok
```

Cette ancienne anomalie etait liee au contexte shell; l'utilisateur a confirme
que le terminal local voit bien la RTX A6000.

Conda existe via:

```text
/home/mohm/miniconda3/bin/conda
```

Le dossier n'est pas encore un repo Git:

```text
fatal: not a git repository
```

## Pipeline actuel a garder

Pipeline fonctionnel retenu:

1. CARLA fournit la simulation, les capteurs RGB/semantic/depth, le trafic, les
   pietons/VRU, les feux/stops et l'expert autopilot.
2. Cosmos Transfer2.5 transforme les videos CARLA en videos/frames
   photorealistes.
3. Alpamayo-1.5 est le VLA/planner actif.
4. `AlpamayoCarlaAdapter` convertit la trajectoire future predite par Alpamayo en
   commandes CARLA `steer/throttle/brake`.
5. Pour ce soir, le meilleur objectif local est un adaptateur rapide
   LoRA-like/residuel qui apprend a mieux convertir trajectoires/etat/instruction
   vers les actions CARLA expert.

Fichiers centraux:

```text
start.sh
environment.yml
scripts/demo.py
scripts/alpamayo_worker.py
scripts/cosmos_transfer_real.py
scripts/prepare_alpamayo_transfer_dataset.py
scripts/run_local_transfer_dataset.sh
scripts/train_local_action_adapter.py
src/models/alpamayo_adapter.py
src/models/local_action_adapter.py
src/data/cosmos_transfer.py
src/carla_env/
external/alpamayo1.5/
external/cosmos-transfer2.5/
docs/CURRENT_PIPELINE.md
docs/STEP2_TRANSFER_DATASET.md
docs/STEP20_ALPAMAYO_CARLA.md
```

Les anciens travaux Qwen/GR00T/Cosmos-Predict doivent rester en backup et ne
plus guider la suite sauf reference ponctuelle.

## Travail effectue depuis l'etape 18 validee

### Etape 18 - modele type GR00T / diffusion

- Implementation d'un pipeline System 2 + System 1:
  - System 2: VLM appele moins souvent, cache toutes les 5 frames.
  - System 1: action head diffusion/DDPM appele a chaque frame.
- Ajout d'un `ActionDiffusionHead` pour remplacer une simple tete MLP.
- Ajout d'un modele `GR00TVLAModel` avec cache VLM et tete diffusion.
- Ajout d'overlay demo indiquant les temps System 1/System 2.
- Entrainement supporte via `--model groot` avec diffusion loss.
- Qwen3VL/Cosmos-Reason2 ont ete testes; bug de tokens image corrige.
- Smoke training passe une fois avec environ:

```text
train_loss=0.909
val_loss=3.181
```

Probleme: en demo, le controle VLA n'etait pas vraiment actif au debut car
l'autopilot CARLA continuait a piloter. Une fois le mode VLA pur active, le
comportement etait mauvais: freinage, a-coups, sorties de voie. Conclusion:
abandonner cette piste comme politique principale.

### Clarification conceptuelle

La diffusion ne remplace pas le VLM. Elle remplace/ameliore la generation des
actions continues. Le VLM reste necessaire pour comprendre la scene, sauf si on
utilise un vrai VLA entraine end-to-end. Cette remarque a motive le passage a
Alpamayo, qui est plus proche d'un planner AV natif.

### Backup premiere version

Les anciens checkpoints, donnees et experiences Qwen/GR00T/Cosmos-Predict ont
ete deplaces dans:

```text
backup_1ere_version/
```

Le backup contient notamment des elements GR00T, Qwen, Cosmos-Predict, anciens
checkpoints et anciennes donnees. Ne pas supprimer.

### Isaac GR00T

- Installation et tests d'Isaac GR00T N1.7.
- Conversion de donnees vers un format GR00T.
- Entrainement probe lance sur `data/groot_carla`.
- Des soucis CUDA_HOME/deepspeed/processor ont ete traites.
- En demo, GR00T restait mauvais pour cette tache CARLA: freinage, a-coups,
  comportement instable, sorties de route.
- Conclusion: GR00T abandonne pour le pipeline actuel.

### Passage a Alpamayo 1.5

Alpamayo est devenu le VLA principal.

Elements ajoutes/configures:

```text
external/alpamayo1.5/
external/alpamayo1.5/a1_5_venv/
scripts/alpamayo_worker.py
src/models/alpamayo_adapter.py
```

Le demo principal reste dans l'environnement CARLA `vla-av-step18`; Alpamayo
tourne dans son propre venv Python 3.12 via un sidecar worker JSONL.

Problemes resolus:

- Mauvais venv active: `cv2`/`pygame` absents si on lance tout depuis le venv
  Alpamayo. Solution: main demo dans `vla-av-step18`, sidecar Alpamayo avec
  `--alpamayo-python external/alpamayo1.5/a1_5_venv/bin/python`.
- Attention implementation: `sdpa` non supporte par Alpamayo dans Transformers.
  Solution: utiliser `attn_implementation="eager"`.
- Commande demo fonctionnelle:

```bash
VLA_AV_CONDA_ENV=vla-av-step18 CARLA_QUALITY=High ./start.sh --real --vla-control \
  --model alpamayo \
  --alpamayo-repo external/alpamayo1.5 \
  --alpamayo-python external/alpamayo1.5/a1_5_venv/bin/python \
  --alpamayo-model-path nvidia/Alpamayo-1.5-10B \
  --lane-assist 0 \
  --spawn-preset traffic_law \
  --camera-width 640 \
  --camera-height 360 \
  --camera-fov 95 \
  --target-speed-kmh 12 \
  --max-vla-throttle 0.30 \
  --instruction "Drive safely, follow the lane, obey red lights and stop signs, yield to pedestrians, cyclists, scooters, and other vehicles."
```

Resultat utilisateur: Alpamayo est moins mauvais que les autres VLA, tient la
ligne environ 20%, mais reste insuffisant pour une demo solide.

## Cosmos Transfer2.5 et dataset

Transfer2.5 a ete valide comme outil de domain transfer. L'utilisateur a note
que certains rendus etaient bons mais que des runs recents etaient parfois flous
ou moins realistes. Les meilleurs resultats venaient de prompts plus precis,
camera hood/capot, resolution plus haute et moins de controles concurrents.

Runs Transfer locaux encore presents:

```text
data/synthetic/transferred_real/transfer25_20260506_155946/
data/synthetic/transferred_real/transfer25_hood_pov_480/
data/synthetic/transferred_real/transfer25_dataset_base_001/
```

Le run `transfer25_dataset_base_001` a permis de creer:

```text
data/alpamayo_transfer_dataset/
```

mais ce dataset ne contient qu'environ 49 frames et doit etre considere comme
smoke/test, pas comme dataset final.

Script principal dataset:

```text
scripts/cosmos_transfer_real.py
```

Il capture:

```text
carla_rgb.mp4
carla_seg.mp4
carla_depth.mp4
episode.jsonl
transfer25_params.json
transfer_output/<run>.mp4
```

`episode.jsonl` contient les labels par frame:

```text
steering
throttle
brake
ego_state
ego_history_xyz
ego_future_xyz
traffic light / stop sign metadata
instruction
paths and timestamps
```

Script manifest Alpamayo:

```text
scripts/prepare_alpamayo_transfer_dataset.py
```

Il extrait les frames photorealistes depuis `transfer_output/*.mp4` et les aligne
avec les labels CARLA dans `manifest.jsonl`.

## Prompts dataset recommandes

Instruction de conduite recommandee:

```text
Drive like a safe autonomous vehicle in an urban environment. Follow the current lane and road markings, keep a smooth centered trajectory, respect speed limits, obey red lights, green lights, stop signs, lane arrows, crosswalks, priority rules, and right-of-way. Yield to pedestrians, cyclists, scooters, motorbikes, parked cars pulling out, and other vehicles. Stop when the path is blocked, wait until it is clear, then continue smoothly without leaving the drivable lane.
```

Prompt visuel Transfer HQ:

```text
sharp forward-facing automotive perception camera view, no visible camera rig, no visible car hood, no dashboard, crisp focus, high detail, stable camera, natural realistic colors, accurate crisp lane markings, continuous lane paint, realistic buildings, realistic vehicles, clean road texture, no cinematic blur, clear daytime urban street, realistic exposure
```

Negative prompt:

```text
CGI, video game, cyberpunk, neon glow, handheld camera, phone in hand, dashcam holder, cartoon, anime, oversaturated colors, distorted buildings, warped lane markings, melted texture, flicker, motion smear, motion blur, blurry, out of focus, soft focus, depth of field blur, compression artifacts, black border, letterbox, pillarbox, low quality
```

## Reprise locale conseillee

### Priorite 0 - reparer/verifier GPU local

GPU local confirme OK par l'utilisateur. Avant un gros run, verifier encore:

```bash
nvidia-smi
conda activate vla-av-step18
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

### Priorite 1 - finaliser nettoyage sans casser le pipeline

Deplacer en backup les donnees de fine-tuning anciennes/non Transfer et les
archives cloud inutiles. Ne pas supprimer:

```text
data/synthetic/transferred/                 -> backup_1ere_version/
data/alpamayo_transfer_dataset/             -> backup_1ere_version/ si on regenere de zero
artifacts/vla-av-b200-project_*.tar.zst     -> backup_1ere_version/
```

Action deja faite localement:

```text
backup_1ere_version/local_cleanup_20260512/synthetic_transferred_legacy/
backup_1ere_version/local_cleanup_20260512/alpamayo_transfer_dataset_smoke/
backup_1ere_version/local_cleanup_20260512/cloud_b200_artifacts/
backup_1ere_version/incomplete_local_runs_20260512/
```

Garder:

```text
data/synthetic/transferred_real/*/transfer_output/
external/alpamayo1.5/
external/cosmos-transfer2.5/
src/models/alpamayo_adapter.py
scripts/cosmos_transfer_real.py
scripts/prepare_alpamayo_transfer_dataset.py
```

### Priorite 2 - smoke dataset local

Script local ajoute avec jauge progression/ETA:

```text
scripts/run_local_transfer_dataset.sh
```

Commencer petit pour verifier qualite/realisme/memoire:

```bash
cd ~/Desktop/vla-av
RUNS=1 FRAMES=49 TRANSFER_RESOLUTION=480 TRANSFER_STEPS=24 \
bash scripts/run_local_transfer_dataset.sh
```

Si GPU local tient bien, monter ensuite a `--transfer-resolution 720` et
`--transfer-num-steps 32` ou `36`. Si OOM, rester a 480/24.

### Priorite 3 - dataset rapide pour ce soir

Objectif realiste local: 20 a 50 clips de 49 frames, soit environ 980 a 2450
frames. Ce n'est pas un gros fine-tuning parfait, mais c'est coherent pour un
adaptateur rapide pret ce soir.

Ensuite:

```bash
python scripts/prepare_alpamayo_transfer_dataset.py \
  --runs-dir data/synthetic/transferred_real \
  --run-glob "transfer25_local_hq_*" \
  --output-dir data/alpamayo_transfer_dataset_local_hq \
  --history-steps 16 \
  --future-steps 64 \
  --dt 0.1 \
  --camera-index 1 \
  --jpeg-quality 97
```

### Priorite 4 - LoRA/adaptateur rapide local

Un vrai LoRA complet Alpamayo-1.5-10B est risque localement pour ce soir:

- repo public surtout oriente inference;
- pas de boucle trainer/loss supervisee prete;
- modele 10B lourd;
- besoin GPU visible et memoire importante.

Chemin recommande pour livrer ce soir:

1. Geler Alpamayo comme planner visuel.
2. Entrainer un petit adaptateur LoRA-like/residuel local qui prend:
   - trajectoire future Alpamayo ou trajectoire expert CARLA,
   - ego speed/history,
   - contexte instruction/metadonnees,
   - et predit une correction ou directement `steer/throttle/brake`.
3. Integrer ce checkpoint dans `AlpamayoCarlaAdapter` avec une option du type:

```text
--action-adapter-checkpoint checkpoints/local_lora_adapter/best.pt
```

Cela cible le vrai point faible observe: la conversion trajectoire -> commandes
CARLA et la stabilite de conduite.

Implementation locale ajoutee:

```text
src/models/local_action_adapter.py
scripts/train_local_action_adapter.py
```

Smoke train deja valide sur le petit manifest de 49 frames:

```text
checkpoints/local_action_adapter_smoke/best.pt
records=49, train=42, val=7
```

Commande d'entrainement sur le futur dataset local:

```bash
python scripts/train_local_action_adapter.py \
  --manifest data/alpamayo_transfer_dataset_local_hq/manifest.jsonl \
  --output-dir checkpoints/local_action_adapter_local_hq \
  --epochs 120 \
  --batch-size 128 \
  --device cuda
```

Activation dans la demo Alpamayo:

```bash
./start.sh --real --vla-control \
  --model alpamayo \
  --alpamayo-action-adapter-checkpoint checkpoints/local_action_adapter_local_hq/best.pt \
  --alpamayo-action-adapter-blend 0.35
```

## GitHub plus tard

Quand le nettoyage local est stable:

```bash
git init
```

Ajouter une `.gitignore` stricte avant le premier commit:

```text
data/
checkpoints/
external/
backup_1ere_version/
artifacts/
*.log
__pycache__/
*.pyc
```

Premier commit recommande:

```bash
git add .
git commit -m "chore: keep active Alpamayo Cosmos CARLA pipeline"
git branch -M main
git remote add origin <URL_GITHUB>
git push -u origin main
```

## Docker plus tard

Dockerisation a faire apres stabilisation locale. A prevoir:

- image runtime CARLA client + demo;
- gestion separee de CARLA simulator lourd;
- volumes pour `data/`, `checkpoints/`, `external/`;
- CUDA base image pour Alpamayo/Cosmos;
- documentation pour lancer CARLA, puis demo, puis generation dataset.

## Decision cle

Le projet ne doit plus chercher a faire marcher toutes les anciennes pistes. La
voie actuelle est:

```text
CARLA expert autopilot labels
-> Cosmos Transfer2.5 photoreal frames
-> Alpamayo planner
-> local fast LoRA-like/action adapter
-> demo CARLA pure VLA control
```

## Reprise cloud 8x B200

Nouvelle opportunite: l'utilisateur dispose a nouveau d'une VM simple d'acces
avec 8 GPU B200. Le process cloud doit reprendre, mais pas sous forme d'un seul
job Cosmos `NUM_GPUS=8`. L'approche retenue est 8 workers independants:

```text
1 GPU B200 = 1 worker = 1 CARLA port = 1 Traffic Manager port = 1 jauge ETA
```

Scripts ajoutes:

```text
cloud_b200/run_parallel_dataset_8gpu.sh
cloud_b200/merge_parallel_dataset.sh
cloud_b200/train_action_adapter_b200.sh
```

Smoke 8 GPU:

```bash
GPUS=8 CLIPS_PER_GPU=1 FRAMES=49 WIDTH=1920 HEIGHT=1080 CAMERA_PRESET=windshield \
TRANSFER_RESOLUTION=720 TRANSFER_STEPS=24 KEEP_INPUT_RESOLUTION=1 \
bash cloud_b200/run_parallel_dataset_8gpu.sh
```

Qualite visee sur B200:

```text
CARLA capture: 1920x1080
Cosmos Transfer native: 720 stable model resolution
Output Transfer/dataset frames: 1920x1080 with KEEP_INPUT_RESOLUTION=1
JPEG extraction: quality 97
```

Gros run propose pour ce soir:

```bash
GPUS=8 CLIPS_PER_GPU=250 FRAMES=49 WIDTH=1920 HEIGHT=1080 CAMERA_PRESET=windshield \
TRANSFER_RESOLUTION=720 TRANSFER_STEPS=48 KEEP_INPUT_RESOLUTION=1 \
bash cloud_b200/run_parallel_dataset_8gpu.sh
```

Cela donne 2000 clips, soit environ 98k frames photorealistes labellisees.

Fusion:

```bash
RUN_PREFIX_BASE=transfer25_b2008_hq \
DATASET_DIR=data/alpamayo_transfer_dataset_b2008_hq_combined \
bash cloud_b200/merge_parallel_dataset.sh
```

Training rapide:

```bash
MANIFEST=data/alpamayo_transfer_dataset_b2008_hq_combined/manifest.jsonl \
OUTPUT_DIR=checkpoints/local_action_adapter_b2008_hq \
EPOCHS=160 BATCH_SIZE=1024 \
bash cloud_b200/train_action_adapter_b200.sh
```

La VM ne partage pas le disque local. Tous les transferts sont faits par SSH:

```bash
# upload local -> VM
rsync -avP -e "ssh -p PORT_VM" artifacts/vla-av-b200-project_*.tar.zst ucloud@ssh.cloud.sdu.dk:~/

# download VM -> local depuis le PC
SSH_PORT=PORT_VM bash cloud_b200/download_results_from_vm.sh
```
