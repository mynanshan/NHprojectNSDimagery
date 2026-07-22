#!/usr/bin/env bash
set -Eeuo pipefail

# Core-NSD data for one subject-specific image-to-brain encoder. The public
# bucket is anonymous, so every AWS command includes --no-sign-request.

BUCKET="s3://natural-scenes-dataset"
DEST="${NSD_DATA_ROOT:-$PWD/data/nsd}"
SUBJECT="01"
SESSION_SPEC="all"
MODE="download"
SKIP_STIMULI=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/download_core_nsd_encoder_mvp.sh [options]' \
    '' \
    '  --subject 01       One subject, 01 through 08 (default: 01)' \
    '  --sessions all     Every completed session (default)' \
    '  --sessions 5       Sessions 1 through 5 (pipeline smoke test)' \
    '  --sessions 1-5     Equivalent explicit range; must begin at 1' \
    '  --dest PATH        Persistent NSD root' \
    '  --estimate         Report large-file sizes without downloading' \
    '  --dry-run          Preview AWS transfers' \
    '  --skip-stimuli     Do not transfer the shared 73K image HDF5' \
    '  -h, --help         Show this help'
}

while (($#)); do
  case "$1" in
    --subject)
      [[ $# -ge 2 ]] || { echo "ERROR: --subject needs a value" >&2; exit 2; }
      SUBJECT="$2"
      shift 2
      ;;
    --sessions)
      [[ $# -ge 2 ]] || { echo "ERROR: --sessions needs a value" >&2; exit 2; }
      SESSION_SPEC="$2"
      shift 2
      ;;
    --dest)
      [[ $# -ge 2 ]] || { echo "ERROR: --dest needs a path" >&2; exit 2; }
      DEST="$2"
      shift 2
      ;;
    --estimate)
      MODE="estimate"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --skip-stimuli)
      SKIP_STIMULI=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is not available in this shell." >&2
  exit 1
}
[[ "$SUBJECT" =~ ^0[1-8]$ ]] || {
  echo "ERROR: subject must be 01 through 08." >&2
  exit 2
}

SESSION_COUNTS=(40 40 32 30 40 32 40 30)
subject_index=$((10#$SUBJECT - 1))
available=${SESSION_COUNTS[$subject_index]}
if [[ "$SESSION_SPEC" == "all" ]]; then
  last_session=$available
elif [[ "$SESSION_SPEC" =~ ^[0-9]+$ ]]; then
  last_session=$((10#$SESSION_SPEC))
elif [[ "$SESSION_SPEC" =~ ^1-([0-9]+)$ ]]; then
  last_session=$((10#${BASH_REMATCH[1]}))
else
  echo "ERROR: --sessions must be all, N, or 1-N." >&2
  exit 2
fi
((last_session >= 1 && last_session <= available)) || {
  echo "ERROR: subj$SUBJECT has 1 through $available completed sessions." >&2
  exit 2
}

AWS_COMMON=(--no-sign-request)
AWS_TRANSFER=()
[[ "$MODE" == "dry-run" ]] && AWS_TRANSFER+=(--dryrun)

human_bytes() {
  awk -v bytes="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", unit, " ");
    i=1;
    while (bytes >= 1024 && i < 5) { bytes /= 1024; i++ }
    printf "%.2f %s", bytes, unit[i]
  }'
}

object_size() {
  local uri="$1"
  local size
  size=$(aws s3 ls "$uri" "${AWS_COMMON[@]}" | awk 'NF >= 4 {print $3; exit}')
  [[ -n "$size" ]] || {
    echo "ERROR: expected NSD object was not found: $uri" >&2
    return 1
  }
  printf '%s' "$size"
}

copy_object() {
  local remote="$1"
  local local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  aws s3 cp "$remote" "$local_path" "${AWS_COMMON[@]}" "${AWS_TRANSFER[@]}"
}

subj="subj$SUBJECT"
beta_remote="$BUCKET/nsddata_betas/ppdata/$subj/func1pt8mm/betas_fithrf_GLMdenoise_RR"
beta_local="$DEST/nsddata_betas/ppdata/$subj/func1pt8mm/betas_fithrf_GLMdenoise_RR"
stimulus_remote="$BUCKET/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
stimulus_local="$DEST/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"

echo "NSD destination: $DEST"
echo "Subject: $subj"
echo "Core sessions: 1-$last_session"
echo "Beta preparation: func1pt8mm/betas_fithrf_GLMdenoise_RR"

if [[ "$MODE" == "estimate" ]]; then
  total=0
  if ((SKIP_STIMULI == 0)); then
    bytes=$(object_size "$stimulus_remote")
    total=$((total + bytes))
    printf '  shared stimulus HDF5: %s\n' "$(human_bytes "$bytes")"
  fi
  for session in $(seq 1 "$last_session"); do
    uri="$beta_remote/betas_session$(printf '%02d' "$session").nii.gz"
    bytes=$(object_size "$uri")
    total=$((total + bytes))
    printf '  session %02d beta: %s\n' "$session" "$(human_bytes "$bytes")"
  done
  printf '  large-file subtotal: %s\n' "$(human_bytes "$total")"
  exit 0
fi

echo
echo "Downloading shared experiment design..."
copy_object \
  "$BUCKET/nsddata/experiments/nsd/nsd_expdesign.mat" \
  "$DEST/nsddata/experiments/nsd/nsd_expdesign.mat"
copy_object \
  "$BUCKET/nsddata/experiments/nsd/nsd_stim_info_merged.csv" \
  "$DEST/nsddata/experiments/nsd/nsd_stim_info_merged.csv"

if ((SKIP_STIMULI == 0)); then
  echo "Downloading the shared 73K image bank..."
  copy_object "$stimulus_remote" "$stimulus_local"
fi

echo "Downloading ROI masks..."
for roi in nsdgeneral prf-visualrois; do
  copy_object \
    "$BUCKET/nsddata/ppdata/$subj/func1pt8mm/roi/${roi}.nii.gz" \
    "$DEST/nsddata/ppdata/$subj/func1pt8mm/roi/${roi}.nii.gz"
done

echo "Downloading beta sessions..."
for session in $(seq 1 "$last_session"); do
  filename="betas_session$(printf '%02d' "$session").nii.gz"
  copy_object "$beta_remote/$filename" "$beta_local/$filename"
done

echo
if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry run complete; no files were downloaded."
else
  echo "Core-NSD encoder download complete."
  echo "Data root: $DEST"
fi
