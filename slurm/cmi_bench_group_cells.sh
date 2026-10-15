#!/usr/bin/env bash
# Shared cell matrix and per-cell launcher for packed MAIA [TBL] benchmark
# groups. Sourced by slurm/bench_group_single.sbatch and the het group wrappers.
#
# A "group" runs the full 12-cell provider/layout/comm matrix for one
# (placement, model, rep) sequentially inside a single allocation:
#   AIX      3 layouts (default/affine/numa) x 3 comms (coll/p2p1/p2pfull) = 9
#   PhyDLL   2 wait modes (default/readiness)                               = 2
#   SmartSim shared database                                                 = 1
#
# Each cell is launched as a child process (the per-cell runners `exit` the
# child only), in its own mktemp run directory, and the group continues on
# failure. Callers must export CMI_AIX_DIR before using the layout mapper.

cmi_group_record() { printf '%s,%s\n' "$2" "$3" >> "$1"; }

# --- model mapping -----------------------------------------------------------
cmi_group_model_file() {
    case "$1" in
        transformer) printf './input/transformer_inference_scripted_fw2.pt\n' ;;
        mlp)         printf './input/mmcp_test_mlp_m5_cpu.pt\n' ;;
        *) printf 'unknown model: %s\n' "$1" >&2; return 1 ;;
    esac
}
cmi_group_model_overlap() { case "$1" in transformer) printf '0\n' ;; mlp) printf '4\n' ;; esac; }
cmi_group_aix_batch()     { case "$1" in transformer) printf '50000\n' ;; mlp) printf '1000000\n' ;; esac; }
cmi_group_phydll_batch()  { case "$1" in transformer) printf '2048\n' ;; mlp) printf '1000000\n' ;; esac; }

# --- AIX layout / comm mapping ----------------------------------------------
cmi_group_layout_lib() {
    case "$1" in
        default) printf '\n' ;;
        affine)  printf '%s/INSTALL-AFFINE\n' "${CMI_AIX_DIR}" ;;
        numa)    printf '%s/INSTALL-NUMA\n' "${CMI_AIX_DIR}" ;;
    esac
}
cmi_group_aix_case()       { case "$1" in collective) printf 'aix-collective\n' ;; p2p1) printf 'aix-p2p1\n' ;; p2pfull) printf 'aix-p2pfull\n' ;; esac; }
cmi_group_aix_comm_short() { case "$1" in collective) printf 'coll\n' ;; p2p1) printf 'p2p1\n' ;; p2pfull) printf 'p2pfull\n' ;; esac; }
cmi_group_aix_het_comm()   { case "$1" in collective) printf 'collective\n' ;; *) printf 'pipelined\n' ;; esac; }
cmi_group_aix_het_credits() {  # $1=comm, $2=total solver tasks
    case "$1" in
        collective) printf '\n' ;;
        p2p1)       printf '1\n' ;;
        p2pfull)    printf '%s\n' "$2" ;;
    esac
}

# --- single-node group -------------------------------------------------------
# $1=model $2=placement $3=np_solver $4=gpus $5=rep $6=root $7=summary
cmi_group_run_single() {
    local model="$1" placement="$2" np_solver="$3" gpus="$4" rep="$5" root="$6" summary="$7"
    local mf ov ab pb lib case_name cell status fails=0
    mf="$(cmi_group_model_file "$model")"
    ov="$(cmi_group_model_overlap "$model")"
    ab="$(cmi_group_aix_batch "$model")"
    pb="$(cmi_group_phydll_batch "$model")"
    local log_dir="${root}/logs/group_cells"
    mkdir -p "$log_dir"
    local layout comm wait

    for layout in default affine numa; do
        lib="$(cmi_group_layout_lib "$layout")"
        for comm in collective p2p1 p2pfull; do
            cell="aix-${placement}-${layout}-$(cmi_group_aix_comm_short "$comm")-${model}-r${rep}"
            case_name="$(cmi_group_aix_case "$comm")"
            status=0
            (
                export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov" MAIA_AIX_BATCHSIZE="$ab"
                export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
                export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0 NP_SOLVER="$np_solver" NP_DL="$gpus"
                export CELL_NAME="$cell" REP="$rep" AIX_LAYOUT_LIB="$lib"
                export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
                bash "${root}/slurm/run_cmi_smoke_gpu.sh" "$case_name"
            ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
            cmi_group_record "$summary" "$cell" "$status"
            (( status == 0 )) || fails=$((fails + 1))
        done
    done

    for wait in default readiness; do
        cell="phydll-${placement}-${wait}-${model}-r${rep}"
        [[ "$wait" == default ]] && case_name=phydll-blocking || case_name=phydll-readiness
        status=0
        (
            export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov" MAIA_PHYDLL_BATCHSIZE="$pb"
            export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
            export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0 NP_SOLVER="$np_solver" NP_DL="$gpus"
            export CELL_NAME="$cell" REP="$rep"
            export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
            bash "${root}/slurm/run_cmi_smoke_gpu.sh" "$case_name"
        ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
        cmi_group_record "$summary" "$cell" "$status"
        (( status == 0 )) || fails=$((fails + 1))
    done

    cell="smartsim-${placement}-shared-${model}-r${rep}"
    status=0
    (
        export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov"
        export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
        export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0 NP_SOLVER="$np_solver"
        export MAIA_SMARTSIM_NUM_GPUS="$gpus" CELL_NAME="$cell" REP="$rep"
        export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
        bash "${root}/slurm/run_cmi_smoke_gpu.sh" smartsim
    ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
    cmi_group_record "$summary" "$cell" "$status"
    (( status == 0 )) || fails=$((fails + 1))

    return "$fails"
}

# --- heterogeneous (surrogate) group ----------------------------------------
# $1=model $2=placement $3=cpu_tasks $4=gpu_tasks $5=gpus $6=rep $7=root $8=summary
cmi_group_run_het() {
    local model="$1" placement="$2" cpu="$3" gpu="$4" gpus="$5" rep="$6" root="$7" summary="$8"
    local mf ov ab pb lib cell status fails=0
    mf="$(cmi_group_model_file "$model")"
    ov="$(cmi_group_model_overlap "$model")"
    ab="$(cmi_group_aix_batch "$model")"
    pb="$(cmi_group_phydll_batch "$model")"
    local total=$((cpu + gpu))
    local log_dir="${root}/logs/group_cells"
    mkdir -p "$log_dir"
    local layout comm wait

    for layout in default affine numa; do
        lib="$(cmi_group_layout_lib "$layout")"
        for comm in collective p2p1 p2pfull; do
            cell="aix-${placement}-${layout}-$(cmi_group_aix_comm_short "$comm")-${model}-r${rep}"
            status=0
            (
                export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
                export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov" MAIA_AIX_BATCHSIZE="$ab"
                export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
                export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0
                export AIX_COMM="$(cmi_group_aix_het_comm "$comm")"
                export AIX_CREDITS="$(cmi_group_aix_het_credits "$comm" "$total")"
                export AIX_LAYOUT_LIB="$lib" CELL_NAME="$cell" REP="$rep"
                export HET_CPU_TASKS="$cpu" HET_GPU_TASKS="$gpu" HET_GPUS="$gpus"
                export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
                bash "${root}/slurm/cmi_bench_het_engine.sh"
            ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
            cmi_group_record "$summary" "$cell" "$status"
            (( status == 0 )) || fails=$((fails + 1))
        done
    done

    for wait in default readiness; do
        cell="phydll-${placement}-${wait}-${model}-r${rep}"
        status=0
        (
            export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
            export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov" MAIA_PHYDLL_BATCHSIZE="$pb"
            export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
            export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0
            export NP_DL="$gpus" PHYDLL_WAIT="$wait"
            export CELL_NAME="$cell" REP="$rep"
            export HET_CPU_TASKS="$cpu" HET_GPU_TASKS="$gpu" HET_GPUS="$gpus"
            export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
            bash "${root}/slurm/cmi_bench_het_engine.sh"
        ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
        cmi_group_record "$summary" "$cell" "$status"
        (( status == 0 )) || fails=$((fails + 1))
    done

    cell="smartsim-${placement}-shared-${model}-r${rep}"
    status=0
    (
        export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
        export PILOT_MODEL_FILE="$mf" ML_CUBE_OVERLAP="$ov"
        export RUN_STEPS="${RUN_STEPS:-200}" SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
        export ML_STEP_TIMING_PRINT=1 DEBUG_EXPORT=0
        export MAIA_SMARTSIM_NUM_GPUS="$gpus" CELL_NAME="$cell" REP="$rep"
        export HET_CPU_TASKS="$cpu" HET_GPU_TASKS="$gpu" HET_GPUS="$gpus"
        export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-300}" READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
        bash "${root}/slurm/cmi_bench_het_engine.sh"
    ) >>"${log_dir}/${cell}.log" 2>&1 || status=$?
    cmi_group_record "$summary" "$cell" "$status"
    (( status == 0 )) || fails=$((fails + 1))

    return "$fails"
}
