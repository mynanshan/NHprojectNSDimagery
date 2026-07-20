#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal NSD-Imagery download for the RSA / representational-geometry MVP.
# The public NSD bucket must be accessed anonymously, hence every AWS command
# below includes --no-sign-request.

BUCKET="s3://natural-scenes-dataset"
DEST="${NSD_DATA_ROOT:-$PWD/data/nsd}"
SUBJECT_SPEC="01"
MODE="download"
INCLUDE_ALLSTIM=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/download_nsdimagery_mvp.sh [options]' \
    '' \
    '  --subjects 01,02   Comma-separated subject numbers (default: 01)' \
    '  --subjects all     All eight subjects' \
    '  --dest PATH        Destination root (default: ./data/nsd or NSD_DATA_ROOT)' \
    '  --estimate         Show sizes of the large beta files; download nothing' \
    '  --dry-run          Show the AWS transfers; download nothing' \
    '  --include-allstim  Also download all 1,149 rendered experiment frames' \
    '  -h, --help         Show this help'
}

# Usage:
#   bash scripts/download_nsdimagery_mvp.sh [options]
#
# Options:
#   --subjects 01,02   Comma-separated subject numbers (default: 01)
#   --subjects all     All eight subjects
#   --dest PATH        Destination root (default: ./data/nsd or NSD_DATA_ROOT)
#   --estimate         Show the sizes of the large beta files; download nothing
#   --dry-run          Show the AWS transfers; download nothing
#   --include-allstim  Also download all 1,149 rendered experiment frames
#   -h, --help         Show this help
#
# Safe first run:
#   bash scripts/download_nsdimagery_mvp.sh --subjects 01 --estimate
#   bash scripts/download_nsdimagery_mvp.sh --subjects 01 --dry-run
#   bash scripts/download_nsdimagery_mvp.sh --subjects 01 --dest /path/with/space
#
# Full MVP after subject 01 has been checked:
#   bash scripts/download_nsdimagery_mvp.sh --subjects all --dest /path/with/space

while (($#)); do
  case "$1" in
    --subjects)
      [[ $# -ge 2 ]] || { echo "ERROR: --subjects needs a value" >&2; exit 2; }
      SUBJECT_SPEC="$2"
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
    --include-allstim)
      INCLUDE_ALLSTIM=1
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

if [[ "$SUBJECT_SPEC" == "all" ]]; then
  SUBJECTS=(01 02 03 04 05 06 07 08)
else
  IFS=',' read -r -a SUBJECTS <<< "$SUBJECT_SPEC"
fi

for subject in "${SUBJECTS[@]}"; do
  [[ "$subject" =~ ^0[1-8]$ ]] || {
    echo "ERROR: subject '$subject' must be 01 through 08 (or use 'all')." >&2
    exit 2
  }
done

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

sync_prefix() {
  local remote="$1"
  local local_dir="$2"
  shift 2
  mkdir -p "$local_dir"
  aws s3 sync "$remote" "$local_dir" "${AWS_COMMON[@]}" "${AWS_TRANSFER[@]}" "$@"
}

echo "NSD destination: $DEST"
echo "Subjects: ${SUBJECTS[*]}"
echo "Preparation: func1pt8mm + nsdimagerybetas_fithrf_GLMdenoise_RR"

if [[ "$MODE" == "estimate" ]]; then
  total=0
  echo
  echo "Large files (the metadata and ROI masks add comparatively little):"
  for subject in "${SUBJECTS[@]}"; do
    uri="$BUCKET/nsddata_betas/ppdata/subj${subject}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5"
    bytes=$(object_size "$uri")
    total=$((total + bytes))
    printf '  subj%s beta: %s\n' "$subject" "$(human_bytes "$bytes")"
  done
  printf '  beta subtotal: %s\n' "$(human_bytes "$total")"
  echo
  echo "Run again without --estimate to download. Use --dry-run first if desired."
  exit 0
fi

echo
echo "Downloading shared experiment metadata and target images..."
sync_prefix \
  "$BUCKET/nsddata/experiments/nsdimagery/" \
  "$DEST/nsddata/experiments/nsdimagery/" \
  --exclude "*" \
  --include "*_dm.mat" \
  --include "designmatrixGLMsingle.mat" \
  --include "*pair_list*" \
  --include "rawtargetimages/*" \
  --include "VVIQ*.pdf"

if ((INCLUDE_ALLSTIM)); then
  echo "Downloading all rendered NSD-Imagery stimulus frames..."
  sync_prefix \
    "$BUCKET/nsddata_stimuli/stimuli/nsdimagery/allstim/" \
    "$DEST/nsddata_stimuli/stimuli/nsdimagery/allstim/"
fi

for subject in "${SUBJECTS[@]}"; do
  subj="subj${subject}"
  echo
  echo "Downloading $subj..."

  sync_prefix \
    "$BUCKET/nsddata/bdata/nsdimagery/" \
    "$DEST/nsddata/bdata/nsdimagery/" \
    --exclude "*" \
    --include "nsdimagery_${subj}_*.tsv"

  copy_object \
    "$BUCKET/nsddata_betas/ppdata/$subj/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5" \
    "$DEST/nsddata_betas/ppdata/$subj/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5"

  for roi in nsdgeneral prf-visualrois streams; do
    copy_object \
      "$BUCKET/nsddata/ppdata/$subj/func1pt8mm/roi/${roi}.nii.gz" \
      "$DEST/nsddata/ppdata/$subj/func1pt8mm/roi/${roi}.nii.gz"
  done

  sync_prefix \
    "$BUCKET/nsddata/freesurfer/$subj/label/" \
    "$DEST/nsddata/freesurfer/$subj/label/" \
    --exclude "*" \
    --include "*prf-visualrois*" \
    --include "*streams*"
done

echo
if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry run complete; no files were downloaded."
else
  echo "Download complete."
  echo "Data root: $DEST"
  echo "Tip: run 'du -sh \"$DEST\"' to check local disk usage."
fi
