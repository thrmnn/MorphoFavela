#!/usr/bin/env bash
# Extend the seasonal solar envelope to one site, generate the SI
# figure, and commit + push.
#
# Usage:
#   bash scripts/shell/extend_solar_envelope_one_site.sh <site_key>
#
# Idempotent: re-running on a site that is already done re-generates
# the gpkg + figure but only commits if there are real diffs.

set -euo pipefail

SITE="${1:-}"
if [[ -z "${SITE}" ]]; then
    echo "Usage: $0 <site_key>" >&2
    exit 2
fi

cd /home/theo/MorphoFavela

echo "============================================================"
echo "  EXTEND SOLAR ENVELOPE — site=${SITE}"
echo "============================================================"

# 1. Compute the seasonal envelope (auto-builds scene.stl if missing).
python scripts/run_street_solar.py --site "${SITE}" --n-jobs -1

# 2. Render the SI figure (combined + 3 standalone panels).
python outputs/paper_figures/figS_solar_envelope.py --site "${SITE}"

# 3. Copy the combined PNG into the report figures dir. Standalones
#    stay in outputs/paper_figures/exports/ for slide re-use; the
#    report's §5.4 only references the combined version per site.
SITE_SAFE="${SITE//ã/a}"
SITE_SAFE="${SITE_SAFE// /_}"
SRC="outputs/paper_figures/exports/figS_solar_envelope_${SITE_SAFE}.png"
DST="docs/technical_report/figures/figS_solar_envelope_${SITE_SAFE}.png"
cp "${SRC}" "${DST}"

# 4. Stage and commit only the figure (the gpkg is gitignored under
#    outputs/) and only if there is something to commit.
git add "${DST}"
if git diff --cached --quiet; then
    echo "No changes to commit for ${SITE}."
    exit 0
fi

git commit -m "feat(solar): extend seasonal envelope to ${SITE_SAFE}

Adds a 4-date (winter/summer solstice + Mar/Sep equinox) solar
envelope on ${SITE_SAFE} via scripts/run_street_solar.py at 30-min
sampling, with the standard SI panel produced by
outputs/paper_figures/figS_solar_envelope.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push
echo "  done: ${SITE}"
