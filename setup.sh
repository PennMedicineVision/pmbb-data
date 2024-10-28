#!/bin/bash

PMBB="${PMBB_DATADIR:-/cbica/projects/pmbb-vision/subjects-102124}"
CACHE_DIR="${PMBB_CACHEDIR:-$HOME/.cache/pmbb}"
mkdir -p $CACHE_DIR

for fn in "patients" "studies" "series" "reports" "scans"; do
  echo "Discovering ${fn}..."
  CACHE_FN=$CACHE_DIR/${fn}.txt
  rm -f $CACHE_FN
  case "$fn" in
    "patients" ) find -P $PMBB -maxdepth 3 -path "*/PMBB*" > $CACHE_FN;;
    "studies" ) find -P $PMBB -maxdepth 4 -path "*/PMBB*/*" > $CACHE_FN;;
    "series" ) find -P $PMBB \
      -mindepth 5 \
      -maxdepth 6 \
      -type d \
      -path "*/PMBB*/*" \
      -not -iname "*slurm*" \
      -not -iname "*loc*" \
      -not -iname "*scout*" \
      > $CACHE_FN;;
    "reports" ) find -P $PMBB \
      -mindepth 5 \
      -maxdepth 6 \
      -type f \
      -path "*/PMBB*/*" \
      -not -iname "*slurm*" \
      -not -iname "*loc*" \
      -not -iname "*scout*" \
      -name "*.txt" \
      > $CACHE_FN;;
    "scans" ) find -P $PMBB \
      -mindepth 5 \
      -maxdepth 6 \
      -type f \
      -not -iname "*slurm*" \
      -not -iname "*loc*" \
      -not -iname "*scout*" \
      -name "*.nii.gz" \
      > $CACHE_FN;;
  esac 
  echo "Wrote ${fn} to ${CACHE_FN}"
done

echo "Done!"
