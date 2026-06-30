#pragma once

// snapshot_writer.hpp — Runtime-conditional HDF5 field snapshot writer
//
// Opt-in via environment variable MAIA_SNAPSHOT_DIR:
//   - If set: snapshots are written to /tmp/maia_snapshots_<SLURM_JOB_ID>.h5
//   - If unset: all operations are no-ops (zero overhead)
//
// After the job finishes, the SLURM script copies the /tmp file to MAIA_SNAPSHOT_DIR.
// Only rank 0 (SLURM_PROCID == "0") writes snapshots.

#include <string>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <highfive/highfive.hpp>

namespace snapshot {

// Module-level state
static bool s_enabled = false;
static bool s_initialized = false;
static std::string s_tmp_filepath;
static std::string s_target_dir;
static HighFive::File* s_file = nullptr;
static int s_snapshot_count = 0;

static bool is_rank_zero() {
    const char* rank = std::getenv("SLURM_PROCID");
    if (rank) {
        return std::string(rank) == "0";
    }
    // Fallback: if not in SLURM, assume single process
    return true;
}

static std::string get_slurm_job_id() {
    const char* job_id = std::getenv("SLURM_JOB_ID");
    if (job_id) {
        return std::string(job_id);
    }
    return "local";
}

/// Initialize the snapshot writer. Call once during solver setup.
/// Checks MAIA_SNAPSHOT_DIR env var; if unset, snapshots are disabled.
static void init() {
    if (s_initialized) return;
    s_initialized = true;

    if (!is_rank_zero()) {
        s_enabled = false;
        return;
    }

    const char* snapshot_dir = std::getenv("MAIA_SNAPSHOT_DIR");
    if (!snapshot_dir) {
        s_enabled = false;
        std::cout << "[snapshot_writer] MAIA_SNAPSHOT_DIR not set, snapshots disabled." << std::endl;
        return;
    }

    s_enabled = true;
    s_target_dir = std::string(snapshot_dir);
    std::string job_id = get_slurm_job_id();
    s_tmp_filepath = "/tmp/maia_snapshots_" + job_id + ".h5";

    std::cout << "[snapshot_writer] Enabled. Writing to: " << s_tmp_filepath << std::endl;
    std::cout << "[snapshot_writer] Target directory (post-job copy): " << s_target_dir << std::endl;

    // Create the HDF5 file (truncate if exists from a previous failed run)
    s_file = new HighFive::File(s_tmp_filepath, HighFive::File::Create | HighFive::File::Truncate);

    // Write metadata
    s_file->createAttribute<std::string>("slurm_job_id", job_id);
    s_file->createAttribute<std::string>("target_dir", s_target_dir);

    s_snapshot_count = 0;
}

/// Format step index as zero-padded string for group name
static std::string step_group_name(int index) {
    std::ostringstream oss;
    oss << "step_" << std::setw(4) << std::setfill('0') << index;
    return oss.str();
}

/// Write a snapshot of U, V, W fields.
///
/// @param uField     Pointer to U velocity field (3D array stored contiguously)
/// @param vField     Pointer to V velocity field
/// @param wField     Pointer to W velocity field
/// @param nCells     Array dimensions [dim0, dim1, dim2]
/// @param nOffsetCells  Offset in each dimension for this rank's partition
/// @param globalTimeStep  The current global time step
/// @param type       "sent" (before ML) or "received" (after ML)
static void write_step(double* uField, double* vField, double* wField,
                       int nCells0, int nCells1, int nCells2,
                       int nOffsetCells0, int nOffsetCells1, int nOffsetCells2,
                       int globalTimeStep, const std::string& type) {
    if (!s_enabled || !s_file) return;

    std::string group_name = step_group_name(s_snapshot_count);
    auto group = s_file->createGroup(group_name);

    // Field dimensions
    std::vector<size_t> dims{(size_t)nCells0, (size_t)nCells1, (size_t)nCells2};

    // Write U, V, W fields
    auto ds_u = group.createDataSet<double>("U", HighFive::DataSpace(dims));
    auto ds_v = group.createDataSet<double>("V", HighFive::DataSpace(dims));
    auto ds_w = group.createDataSet<double>("W", HighFive::DataSpace(dims));

    ds_u.write_raw(uField);
    ds_v.write_raw(vField);
    ds_w.write_raw(wField);

    // Write metadata for this step
    std::vector<int> nCells_vec = {nCells0, nCells1, nCells2};
    std::vector<int> nOffsetCells_vec = {nOffsetCells0, nOffsetCells1, nOffsetCells2};

    group.createDataSet<int>("nCells", HighFive::DataSpace({3})).write(nCells_vec);
    group.createDataSet<int>("nOffsetCells", HighFive::DataSpace({3})).write(nOffsetCells_vec);

    group.createAttribute<int>("globalTimeStep", globalTimeStep);
    group.createAttribute<std::string>("type", type);

    s_snapshot_count++;

    // Flush periodically to avoid data loss on crash
    if (s_snapshot_count % 10 == 0) {
        s_file->flush();
        std::cout << "[snapshot_writer] Flushed after " << s_snapshot_count << " snapshots." << std::endl;
    }
}

/// Finalize the snapshot writer. Call once during solver teardown.
static void finalize() {
    if (!s_enabled || !s_file) return;

    // Write total snapshot count as root attribute
    s_file->createAttribute<int>("total_snapshots", s_snapshot_count);

    std::cout << "[snapshot_writer] Finalizing. Total snapshots written: " << s_snapshot_count << std::endl;
    std::cout << "[snapshot_writer] File: " << s_tmp_filepath << std::endl;

    delete s_file;
    s_file = nullptr;
    s_enabled = false;
}

} // namespace snapshot
