#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAIL=0
MAX_BYTES=$((95 * 1024 * 1024))
SECRET_PATTERN='(hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----)'

fail() {
  printf '[publish-audit] FAIL: %s\n' "$1" >&2
  FAIL=1
}

while IFS= read -r -d '' path; do
  base="${path##*/}"
  case "$base" in
    .env|.env.*|credentials.json|secrets.json|id_rsa|id_ed25519|*.pem|*.key)
      fail "credential-like filename is tracked: $path"
      ;;
  esac
done < <(git ls-files -z)

# Batch object-size lookup avoids spawning one Git process for each vendored
# file. LFS-managed binaries appear here as small pointer blobs, as intended.
LARGE_OBJECTS="$({
  git ls-files -s | awk '{print $2}' | sort -u \
    | git cat-file --batch-check='%(objectname) %(objectsize)' \
    | awk -v maximum="$MAX_BYTES" '$2 > maximum {print $1 " " $2}'
} || true)"
if [[ -n "$LARGE_OBJECTS" ]]; then
  while read -r object size; do
    paths="$(git ls-files -s | awk -v wanted="$object" '$2 == wanted {sub(/^[^\t]*\t/, ""); print}')"
    fail "Git blob exceeds 95 MiB (use LFS or ignore it): ${paths:-$object} ($size bytes)"
  done <<< "$LARGE_OBJECTS"
fi

SECRET_FILES="$(git grep --cached -I -l -E "$SECRET_PATTERN" -- . 2>/dev/null || true)"
if [[ -n "$SECRET_FILES" ]]; then
  while IFS= read -r path; do
    fail "token/private-key pattern found in tracked content: $path"
  done <<< "$SECRET_FILES"
fi

if git ls-files | grep -Eq '(^|/)\.streamlit/(secrets|credentials)\.toml$'; then
  fail "Streamlit secret file is tracked"
fi

if [[ -f streamlit_share/dashboard_snapshot.html ]]; then
  if grep -aEq "$SECRET_PATTERN" streamlit_share/dashboard_snapshot.html; then
    fail "secret pattern found in Streamlit snapshot"
  fi
  if ! grep -q 'Read-only dashboard' streamlit_share/dashboard_snapshot.html; then
    fail "Streamlit snapshot does not advertise read-only mode"
  fi
fi

if (( FAIL != 0 )); then
  exit 1
fi

FILES="$(git ls-files | wc -l)"
echo "[publish-audit] OK: $FILES tracked files checked; no known secret pattern or oversized Git blob found."
