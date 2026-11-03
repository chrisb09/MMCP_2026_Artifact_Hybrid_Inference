#!/usr/bin/env bash
# One isolated correctness-export pilot. Default is a no-write dry run.
set -euo pipefail
root="$(realpath "$(dirname "$0")/..")"
mode="${1:---dry-run}"
[[ $# -le 1 && ( "$mode" == --dry-run || "$mode" == --submit ) ]] || {
    printf 'Usage: bash %s [--dry-run|--submit]\n' "$0" >&2; exit 2;
}
base="${PHYDLL_PILOT_ROOT:-${root}/scratch/phydll_cap50000_pilots}"
if [[ "$mode" == --submit ]]; then
    mkdir -p -- "$base"
    campaign="$(mktemp -d "$(realpath "$base")/pilot_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    mkdir "$campaign/scratch"
else
    campaign="${base}/pilot_<unique>"
fi
exports="ALL,PROJECT_DIR=${root},SMOKE_SCRATCH_ROOT=${campaign}/scratch,PILOT_MODEL_FILE=./input/transformer_inference_scripted_fw2.pt,ML_CUBE_OVERLAP=0,MAIA_PHYDLL_BATCHSIZE=50000,RUN_STEPS=200,SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,MAIA_ML_WORKLOAD_LOG=1,MAIA_ML_STEP_TIMING_SYNC=1,DEBUG_EXPORT=1,NP_SOLVER=24,NP_DL=1,CASE_TIMEOUT_SECONDS=600,CELL_NAME=phydll-24c1g-default-transformer-cap50000-pilot"
[[ "$root$campaign" != *','* && "$root$campaign" != *$'\n'* ]] || { printf 'Paths cannot contain commas or newlines.\n' >&2; exit 2; }
cmd=(sbatch --parsable --partition=c23g --account=rwth0792 --gres=gpu:1
    --exclusive --mem=0 --nodes=1 --ntasks=24 --cpus-per-task=1 --time=00:20:00
    --job-name=phydll-cap50000-pilot --output="${campaign}/slurm_%j.out"
    --error="${campaign}/slurm_%j.err" --export="$exports"
    "${root}/slurm/run_cmi_smoke_gpu.sh" phydll-blocking)
printf 'Model: %s\nCampaign: %s\n' "$root/input/transformer_inference_scripted_fw2.pt" "$campaign"
printf 'Clean environment launch: env -i HOME=%q USER=%q PATH=%q BASH_ENV=/etc/profile ' "${HOME:-}" "${USER:-}" "$PATH"
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$mode" == --dry-run ]]; then
    printf 'Dry run: no directories created and no job submitted.\n'
    exit 0
fi
{
    printf 'campaign=%s\nmodel=%s\ncase=phydll-blocking\ncap=50000\n' "$campaign" "$root/input/transformer_inference_scripted_fw2.pt"
    printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
    for repo in "$root" "$root/maia" "$root/CPP-ML-Interface" "$root/CPP-ML-Interface/extern/phydll"; do
        printf '\nrepository=%s\n' "$repo"
        git -C "$repo" rev-parse HEAD 2>&1 || true
        git -C "$repo" status --short 2>&1 || true
    done
    for f in "$root/input/transformer_inference_scripted_fw2.pt" "$root/input/properties_run_les_ref_medium.toml" "$root/config_phydll.toml" "$root/slurm/run_cmi_smoke_gpu.sh" "$root/maia/build_gnu_production/bin/maia" "$root/maia/build_gnu_production/CPP-ML-Interface/dl_clients/phydll_dl_client" "$root/maia/build_gnu_production/bin/phydll_dl_client"; do
        if [[ -f "$f" ]]; then sha256sum "$f"; else printf 'missing=%s\n' "$f"; fi
    done
} > "$campaign/manifest.txt"
for entry in root:. maia:maia cmi:CPP-ML-Interface phydll:CPP-ML-Interface/extern/phydll; do
    name="${entry%%:*}"; repo="${entry#*:}"
    git -C "$root/$repo" diff --binary HEAD > "$campaign/${name}.diff" 2> "$campaign/${name}.diff.err" || true
done
# Do not let caller overrides silently change placement, model or runtime knobs.
if jid="$(env -i HOME="${HOME:-}" USER="${USER:-}" PATH="$PATH" BASH_ENV=/etc/profile "${cmd[@]}")"; then
    printf 'sbatch_response=%s\n' "$jid" >> "$campaign/manifest.txt"
    printf 'Submitted: %s\nManifest: %s/manifest.txt\n' "$jid" "$campaign"
else
    status=$?
    printf 'sbatch_exit_status=%s\n' "$status" >> "$campaign/manifest.txt"
    printf 'Submission failed; evidence: %s/manifest.txt\n' "$campaign" >&2
    exit "$status"
fi
