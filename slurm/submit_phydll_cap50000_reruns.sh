#!/usr/bin/env bash
# Isolated production submitter: 24 PhyDLL Transformer runs at cap 50000.
# Models x placements x wait modes x repeats.
#
# Usage: bash slurm/submit_phydll_cap50000_reruns.sh [--dry-run|--submit]
#   --dry-run  (default) print what would be submitted; creates no directories.
#   --submit   create campaign directory, write provenance, submit via sbatch.
#
# Override matrix via env, e.g.:
#   MODELS=transformer PLACEMENTS='24c1g' WAITS=default REPS=1 \
#     bash slurm/submit_phydll_cap50000_reruns.sh --dry-run
set -euo pipefail

cd "$(dirname "$0")/.."
root="$(pwd)"

mode="${1:---dry-run}"
[[ $# -le 1 && ( "$mode" == --dry-run || "$mode" == --submit ) ]] || {
    printf 'Usage: bash %s [--dry-run|--submit]\n' "$0" >&2; exit 2
}

# ---------------------------------------------------------------------------
# Matrix knobs
# ---------------------------------------------------------------------------
MODELS="${MODELS:-transformer}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
WAITS="${WAITS:-default readiness}"
REPS="${REPS:-3}"

# ---------------------------------------------------------------------------
# Validate matrix values before doing anything else
# ---------------------------------------------------------------------------
for _m in ${MODELS}; do
    case "${_m}" in transformer|mlp) ;; *)
        printf 'ERROR: unknown model %q (must be transformer or mlp)\n' "${_m}" >&2; exit 2 ;;
    esac
done
for _p in ${PLACEMENTS}; do
    case "${_p}" in 24c1g|96c4g|24p24c1g|96p96c4g) ;; *)
        printf 'ERROR: unknown placement %q (must be 24c1g, 96c4g, 24p24c1g, or 96p96c4g)\n' "${_p}" >&2; exit 2 ;;
    esac
done
for _w in ${WAITS}; do
    case "${_w}" in default|readiness) ;; *)
        printf 'ERROR: unknown wait %q (must be default or readiness)\n' "${_w}" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Campaign directory (only created on --submit)
# ---------------------------------------------------------------------------
campaign_base="${PHYDLL_RERUN_CAMPAIGN_BASE:-${root}/results/phydll_cap50000_reruns}"
if [[ "$mode" == --submit ]]; then
    mkdir -p -- "${campaign_base}" "${root}/logs" "${root}/scratch/phydll_cap50000_reruns"
    campaign="$(mktemp -d "${campaign_base}/campaign_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
else
    campaign="${campaign_base}/campaign_<unique>"
fi

# Guard against commas and newlines in paths used inside --export=
[[ "${root}${campaign}" != *','* && "${root}${campaign}" != *$'\n'* ]] || {
    printf 'ERROR: Paths cannot contain commas or newlines.\n' >&2; exit 2
}

manifest_csv="${campaign}/manifest.csv"

# ---------------------------------------------------------------------------
# Per-model helpers
# ---------------------------------------------------------------------------
model_file()        { case "$1" in transformer) printf '%s' "./input/transformer_inference_scripted_fw2.pt" ;; mlp) printf '%s' "./input/mmcp_test_mlp_m5_cpu.pt" ;; esac; }
model_overlap()     { case "$1" in transformer) printf '%s' "0" ;; mlp) printf '%s' "4" ;; esac; }
model_cap()         { case "$1" in transformer) printf '%s' "50000" ;; mlp) printf '%s' "1000000" ;; esac; }

# ---------------------------------------------------------------------------
# Provenance (--submit only)
# ---------------------------------------------------------------------------
write_provenance() {
    local prov="${campaign}/provenance.txt"
    printf '# PhyDLL cap50000 reruns provenance\ncampaign=%s\ndate_utc=%s\n\n' \
        "${campaign}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${prov}"
    for repo in "${root}" "${root}/maia" "${root}/CPP-ML-Interface" \
                "${root}/CPP-ML-Interface/extern/phydll"; do
        printf '\n[repository: %s]\n' "${repo}" >>"${prov}"
        git -C "${repo}" rev-parse HEAD >>"${prov}" 2>&1 || true
        git -C "${repo}" status --short  >>"${prov}" 2>&1 || true
    done
    printf '\n[source diffs]\n' >>"${prov}"
    for entry in root:. maia:maia cmi:CPP-ML-Interface phydll:CPP-ML-Interface/extern/phydll; do
        name="${entry%%:*}"; repo="${entry#*:}"
        git -C "${root}/${repo}" diff --binary HEAD \
            >"${campaign}/${name}.diff" 2>"${campaign}/${name}.diff.err" || true
        printf 'diff: %s\n' "${campaign}/${name}.diff" >>"${prov}"
    done
    printf '\n[input file sha256]\n' >>"${prov}"
    for f in \
        "${root}/input/transformer_inference_scripted_fw2.pt" \
        "${root}/input/mmcp_test_mlp_m5_cpu.pt" \
        "${root}/input/properties_run_les_ref_medium.toml" \
        "${root}/config_phydll.toml" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" \
        "${root}/slurm/bench_het_24p24c1g.sbatch" \
        "${root}/slurm/bench_het_96p96c4g.sbatch" \
        "${root}/maia/build_gnu_production/bin/maia" \
        "${root}/maia/build_gnu_production/CPP-ML-Interface/dl_clients/phydll_dl_client" \
        "${root}/maia/build_gnu_production/bin/phydll_dl_client"; do
        if [[ -f "${f}" ]]; then sha256sum "${f}" >>"${prov}"; else printf 'missing=%s\n' "${f}" >>"${prov}"; fi
    done
}

# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

# Write one manifest CSV row (append).
record_manifest() {
    local cell="$1" jobid="$2" placement="$3" model="$4" wait="$5" rep="$6" \
          cap="$7" status_dir="$8" revision="$9" dry_run="${10}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cell}" "${jobid}" "${placement}" "${model}" "${wait}" "${rep}" \
        "${cap}" "${status_dir}" "${revision}" "${dry_run}" \
        >>"${manifest_csv}"
}

submit_single() {
    local placement="$1" model="$2" wait="$3" rep="$4"
    local gres ntasks npdl case_name
    case "${placement}" in
        24c1g) gres=1; ntasks=24; npdl=1 ;;
        96c4g) gres=4; ntasks=96; npdl=4 ;;
        *) return 1 ;;
    esac
    case "${wait}" in
        default)   case_name=phydll-blocking  ;;
        readiness) case_name=phydll-readiness ;;
    esac
    local cap
    cap="$(model_cap "${model}")"
    local cell="phydll-${placement}-${wait}-${model}-cap50000-r${rep}"
    local exports
    exports="ALL,PROJECT_DIR=${root},SMOKE_SCRATCH_ROOT=${root}/scratch/phydll_cap50000_reruns"
    exports="${exports},PILOT_MODEL_FILE=$(model_file "${model}")"
    exports="${exports},ML_CUBE_OVERLAP=$(model_overlap "${model}")"
    exports="${exports},MAIA_PHYDLL_BATCHSIZE=${cap}"
    exports="${exports},DEBUG_EXPORT=0,RUN_STEPS=200,SOLUTION_INTERVAL=10000"
    exports="${exports},ML_STEP_TIMING_PRINT=1,MAIA_ML_WORKLOAD_LOG=1"
    exports="${exports},CASE_TIMEOUT_SECONDS=600"
    exports="${exports},NP_SOLVER=${ntasks},NP_DL=${npdl}"
    exports="${exports},REP=${rep},CELL_NAME=${cell}"
    local cmd
    cmd=(sbatch --parsable
        --partition=c23g --account=rwth0792 --gres="gpu:${gres}"
        --exclusive --mem=0 --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1
        --time=00:20:00
        --job-name="cmi-${cell}"
        --export="${exports}"
        "${root}/slurm/run_cmi_smoke_gpu.sh" "${case_name}")

    if [[ "$mode" == --dry-run ]]; then
        printf 'DRY_RUN cell=%s  cap=%s  placement=%s  wait=%s  rep=%s\n' \
            "${cell}" "${cap}" "${placement}" "${wait}" "${rep}"
        printf '  cmd: env -i HOME=%q USER=%q PATH=%q BASH_ENV=/etc/profile ' \
            "${HOME:-}" "${USER:-}" "$PATH"
        printf '%q ' "${cmd[@]}"; printf '\n'
        record_manifest "${cell}" "DRY_RUN" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "" "" "1"
        return 0
    fi

    local jid status_dir
    if jid="$(env -i HOME="${HOME:-}" USER="${USER:-}" PATH="$PATH" BASH_ENV=/etc/profile \
               "${cmd[@]}")"; then
        status_dir="${root}/scratch/phydll_cap50000_reruns/${cell}_${jid}"
        record_manifest "${cell}" "${jid}" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "${status_dir}" \
            "$(git -C "${root}" rev-parse --short HEAD 2>/dev/null || echo unknown)" "0"
        printf 'Submitted: %s  jobid=%s\n' "${cell}" "${jid}"
    else
        local exitcode=$?
        printf 'ERROR: sbatch failed for %s (exit %s)\n' "${cell}" "${exitcode}" >&2
        record_manifest "${cell}" "FAILED" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "" "" "0"
        return "${exitcode}"
    fi
}

submit_het() {
    local placement="$1" model="$2" wait="$3" rep="$4"
    local script cpu gpu gpus npdl
    case "${placement}" in
        24p24c1g) script=bench_het_24p24c1g.sbatch; cpu=24; gpu=24; gpus=1; npdl=1 ;;
        96p96c4g) script=bench_het_96p96c4g.sbatch; cpu=96; gpu=96; gpus=4; npdl=4 ;;
        *) return 1 ;;
    esac
    local cap
    cap="$(model_cap "${model}")"
    local cell="phydll-${placement}-${wait}-${model}-cap50000-r${rep}"
    local exports
    exports="ALL,PROJECT_DIR=${root},SMOKE_SCRATCH_ROOT=${root}/scratch/phydll_cap50000_reruns"
    exports="${exports},PILOT_MODEL_FILE=$(model_file "${model}")"
    exports="${exports},ML_CUBE_OVERLAP=$(model_overlap "${model}")"
    exports="${exports},MAIA_PHYDLL_BATCHSIZE=${cap}"
    exports="${exports},DEBUG_EXPORT=0,RUN_STEPS=200,SOLUTION_INTERVAL=10000"
    exports="${exports},ML_STEP_TIMING_PRINT=1,MAIA_ML_WORKLOAD_LOG=1"
    exports="${exports},CASE_TIMEOUT_SECONDS=600"
    exports="${exports},CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL"
    exports="${exports},NP_DL=${npdl},PHYDLL_WAIT=${wait}"
    exports="${exports},REP=${rep},CELL_NAME=${cell}"
    local cmd
    cmd=(sbatch --parsable
        --export="${exports}"
        "${root}/slurm/${script}")

    if [[ "$mode" == --dry-run ]]; then
        printf 'DRY_RUN cell=%s  cap=%s  placement=%s  wait=%s  rep=%s\n' \
            "${cell}" "${cap}" "${placement}" "${wait}" "${rep}"
        printf '  cmd: env -i HOME=%q USER=%q PATH=%q BASH_ENV=/etc/profile ' \
            "${HOME:-}" "${USER:-}" "$PATH"
        printf '%q ' "${cmd[@]}"; printf '\n'
        record_manifest "${cell}" "DRY_RUN" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "" "" "1"
        return 0
    fi

    local jid status_dir
    if jid="$(env -i HOME="${HOME:-}" USER="${USER:-}" PATH="$PATH" BASH_ENV=/etc/profile \
               "${cmd[@]}")"; then
        status_dir="${root}/scratch/phydll_cap50000_reruns/${cell}_${jid}"
        record_manifest "${cell}" "${jid}" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "${status_dir}" \
            "$(git -C "${root}" rev-parse --short HEAD 2>/dev/null || echo unknown)" "0"
        printf 'Submitted: %s  jobid=%s\n' "${cell}" "${jid}"
    else
        local exitcode=$?
        printf 'ERROR: sbatch failed for %s (exit %s)\n' "${cell}" "${exitcode}" >&2
        record_manifest "${cell}" "FAILED" "${placement}" "${model}" "${wait}" \
            "${rep}" "${cap}" "" "" "0"
        return "${exitcode}"
    fi
}

# ---------------------------------------------------------------------------
# Announce and write headers
# ---------------------------------------------------------------------------
printf 'PhyDLL cap50000 reruns submitter\n'
printf '  mode       : %s\n' "${mode}"
printf '  MODELS     : %s\n' "${MODELS}"
printf '  PLACEMENTS : %s\n' "${PLACEMENTS}"
printf '  WAITS      : %s\n' "${WAITS}"
printf '  REPS       : %s\n' "${REPS}"
printf '  campaign   : %s\n' "${campaign}"

if [[ "$mode" == --submit ]]; then
    # Write CSV header
    printf 'cell,jobid,placement,model,wait,rep,cap,status_dir,revision,dry_run\n' \
        >"${manifest_csv}"
    write_provenance
    printf '  provenance : %s/provenance.txt\n' "${campaign}"
else
    # In dry-run, write CSV header to a temp location not persisted on disk.
    # We still call record_manifest so we can use it uniformly; redirect to /dev/null.
    manifest_csv=/dev/null
fi

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for wait in ${WAITS}; do
            for rep in $(seq 1 "${REPS}"); do
                case "${placement}" in
                    24c1g|96c4g)      submit_single "${placement}" "${model}" "${wait}" "${rep}" ;;
                    24p24c1g|96p96c4g) submit_het   "${placement}" "${model}" "${wait}" "${rep}" ;;
                esac
            done
        done
    done
done

if [[ "$mode" == --submit ]]; then
    printf 'Campaign complete.\n  manifest : %s\n' "${manifest_csv}"
else
    printf 'Dry run complete. No directories created, no jobs submitted.\n'
fi
