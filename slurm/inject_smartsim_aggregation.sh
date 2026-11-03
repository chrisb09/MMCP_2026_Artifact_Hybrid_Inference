#!/usr/bin/env bash
# Sourced helper. Apply aggregation settings to the run-local [library] table
# and report them only after a successful rewrite. Batch size zero disables
# aggregation and writes zeros for all three settings.

_ssai_inject() {
    local batch="${SMARTSIM_BATCH_SIZE-}" min_batch="${SMARTSIM_MIN_BATCH_SIZE-}"
    local timeout="${SMARTSIM_MIN_BATCH_TIMEOUT_MS-}" value config tmp enabled

    for value in "$batch" "$min_batch" "$timeout"; do
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf 'inject_smartsim_aggregation: all three settings must be non-negative integers.\n' >&2
            return 2
        fi
    done
    # Normalize decimal input before arithmetic and reject values outside the
    # signed integer range used by SmartRedis (also avoids shell overflow).
    for value in batch min_batch timeout; do
        while [[ ${!value} == 0?* ]]; do printf -v "$value" '%s' "${!value:1}"; done
        local number="${!value}"
        if (( ${#number} > 10 )) || { (( ${#number} == 10 )) && [[ "$number" > 2147483647 ]]; }; then
            printf 'inject_smartsim_aggregation: setting exceeds SmartRedis integer range: %s\n' "$number" >&2
            return 2
        fi
    done

    enabled=1
    if (( batch == 0 )); then
        enabled=0
        min_batch=0
        timeout=0
    elif (( min_batch == 0 || min_batch > batch || timeout == 0 )); then
        printf 'inject_smartsim_aggregation: enabled batching requires 0 < min_batch_size <= batch_size and timeout > 0.\n' >&2
        return 2
    fi

    config="${SMARTSIM_CONFIG:-./config_smartsim.toml}"
    if [[ ! -f "$config" || -L "$config" ]]; then
        printf 'inject_smartsim_aggregation: config must be a regular non-symlink file: %s\n' "$config" >&2
        return 2
    fi
    tmp=$(mktemp "${config}.XXXXXX") || return 2
    if ! awk -v batch="$batch" -v min_batch="$min_batch" -v timeout="$timeout" '
        /^[[:space:]]*\[/ {
            in_library = ($0 ~ /^[[:space:]]*\[library\][[:space:]]*(#.*)?$/)
            if (in_library) sections++
        }
        in_library && /^[[:space:]]*(batch_size|min_batch_size|min_batch_timeout)[[:space:]]*=/ {
            key = $0
            sub(/^[[:space:]]*/, "", key)
            sub(/[[:space:]]*=.*/, "", key)
            count[key]++
            if ($0 !~ /^[[:space:]]*(batch_size|min_batch_size|min_batch_timeout)[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*(#.*)?$/) invalid = 1
            sub(/=.*/, "= " (key == "batch_size" ? batch : key == "min_batch_size" ? min_batch : timeout))
        }
        { print }
        END {
            if (invalid || sections != 1 || count["batch_size"] != 1 || count["min_batch_size"] != 1 || count["min_batch_timeout"] != 1) {
                print "inject_smartsim_aggregation: expected one [library] table with exactly one of each batch field" > "/dev/stderr"
                exit 2
            }
        }
    ' "$config" > "$tmp" || ! chmod --reference="$config" "$tmp" || ! mv -f -- "$tmp" "$config"; then
        rm -f -- "$tmp"
        return 2
    fi

    local report
    printf -v report 'SMARTSIM_AGGREGATION_ENABLED=%s\nSMARTSIM_BATCH_SIZE_EFFECTIVE=%s\nSMARTSIM_MIN_BATCH_SIZE_EFFECTIVE=%s\nSMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE=%s\n' \
        "$enabled" "$batch" "$min_batch" "$timeout"
    if [[ -f run.info ]]; then
        printf '%s\n' "$report" >> run.info || return 2
    fi
    printf '%s\n' "$report"
}

_ssai_inject
_ssai_status=$?
unset -f _ssai_inject
if (( _ssai_status != 0 )); then
    unset _ssai_status
    return 2
fi
unset _ssai_status
