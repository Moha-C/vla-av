# GitHub Setup For This Project

This folder is the active project root:

```bash
cd ~/Desktop/vla-av
```

## 1. Verify What Will Be Committed

The project contains huge local artifacts. Before the first commit, verify ignored
files:

```bash
git status --ignored
```

Expected ignored heavy folders include:

```text
logs/
exports/
vm_backups/
backup_1ere_version/
models/
pretrained/
.cache/
.conda_node/
experiments/TwinSentinel_Project/maps/
experiments/TwinSentinel_Project/runs/
```

## 2. Fix The Broken Root .git Folder

At the time this file was written, `~/Desktop/vla-av/.git` exists but is empty
and not a valid Git repository. Rename it once:

```bash
cd ~/Desktop/vla-av
mv .git .git_broken_empty_$(date +%Y%m%d_%H%M%S)
git init
```

If `git status` already works on your machine, do not run the `mv` line.

## 3. Decide How To Handle Nested Repositories

Some folders were cloned from other repositories and contain their own `.git`.
For a single private GitHub repo with all source files inside, flatten them:

```bash
find external experiments -mindepth 2 -maxdepth 3 -type d -name .git -print
```

Then move the nested `.git` folders out of the tree or delete them only after
you are sure you do not need their separate history.

Safer option:

```bash
mkdir -p ~/Desktop/vla-av-nested-git-backups
find external experiments -mindepth 2 -maxdepth 3 -type d -name .git -exec bash -c '
  for d do
    parent="$(dirname "$d")"
    safe="$(echo "$parent" | tr "/" "_")"
    mv "$d" "$HOME/Desktop/vla-av-nested-git-backups/${safe}.git"
  done
' bash {} +
```

## 4. Configure Git LFS

```bash
git lfs install
git lfs track "*.pt" "*.ckpt" "*.safetensors" "*.onnx"
git add .gitattributes
```

The main SimLingo Hugging Face checkpoint and pretrained VLM folders are not
tracked by default. Use `scripts/download_simlingo_model.sh` after cloning.

## 5. First Commit

```bash
git add .gitignore .gitattributes README.md requirements.txt requirements.freeze.txt environment.simlingo.yml environment.simlingo-lock.yml docs scripts src external/simlingo experiments/dreamer_ppo_carla experiments/dreamer_ppo_carla_sdbs_fresh experiments/TwinSentinel_Project
git status --short
git commit -m "Initial VLA-AV SimLingo Dreamer SUMO pipeline"
```

If Git reports huge files, remove them from the index before committing:

```bash
git restore --staged path/to/huge/file
```

## 6. Create A Private GitHub Repo And Push

With GitHub CLI:

```bash
gh auth login
gh repo create vla-av --private --source=. --remote=origin --push
```

Manual remote:

```bash
git remote add origin git@github.com:<your-user>/vla-av.git
git branch -M main
git push -u origin main
```

## 7. Fresh Clone Checklist

After cloning on another machine:

```bash
cd vla-av
bash scripts/install_system_deps_ubuntu22.sh
conda env create -f environment.simlingo.yml
conda activate simlingo
bash scripts/download_simlingo_model.sh
export CARLA_ROOT=$HOME/carla_simulator
export SUMO_HOME=/usr/share/sumo
bash scripts/run_simlingo_dashboard.sh
```

CARLA 0.9.15 itself must be installed separately at `$CARLA_ROOT`.
