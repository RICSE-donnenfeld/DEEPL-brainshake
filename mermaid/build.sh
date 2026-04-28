#!/usr/bin/env bash
set -euo pipefail

for f in mermaid/*.mmd; do
	name=$(basename "${f%.mmd}")
	echo "Building $name..."
	mmdc -i "$f" -o "out/mermaid/${name}.svg" --backgroundColor transparent --theme dark
done
