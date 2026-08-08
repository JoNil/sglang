#!/bin/sh
set -eu

seed=/opt/sglang-thor/flashinfer-cache-seed
target=/root/.cache/flashinfer

if [ ! -d "$seed" ]; then
  exit 0
fi

mkdir -p "$target"
cp -a "$seed/." "$target/"
echo "Seeded Thor FlashInfer W4/W8 cache"
