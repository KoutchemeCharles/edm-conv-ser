#!/usr/bin/env bash
set -uo pipefail   # note: NO -e

pids=()
names=()

bash scripts/bash/new/models/qwen_falcon.sh & pids+=($!) names+=("qweni")
bash scripts/bash/new/models/qwenb_falcon.sh & pids+=($!) names+=("qwenb")
bash scripts/bash/new/models/gpt5.sh & pids+=($!) names+=("gpt5")

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "❌ ${names[$i]} failed"
    fail=1
  else
    echo "✅ ${names[$i]} finished"
  fi
done

exit "$fail"
