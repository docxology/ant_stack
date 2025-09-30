#!/usr/bin/env bash
set -euo pipefail
LINK_COLOR=${LINK_COLOR:-blue}
MERMAID_STRATEGY=${MERMAID_STRATEGY:-none}
MERMAID_IMG_FORMAT=${MERMAID_IMG_FORMAT:-png}
bash tools/render_pdf.sh ant_stack
bash tools/render_pdf.sh complexity_energetics
echo "Built: 1_ant_stack.pdf 2_complexity_energetics.pdf"

