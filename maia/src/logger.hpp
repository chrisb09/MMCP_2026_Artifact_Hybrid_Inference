#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <span>

#include <iostream>
#include <array>
#include <cstdlib>
#include <vector>


std::vector<long long> step_times;

// We want to log to a file

// Ideally, we want to use slurm job info to name the log file accordingly

// Let's get slurm job id from environment variable
std::string get_slurm_job_id() {
    const char* job_id = std::getenv("SLURM_JOB_ID");
    if (job_id) {
        return std::string(job_id);
    } else {
        return "no_slurm_job";
    }
}

std::string get_tasks_per_node() {
    const char* tasks_per_node = std::getenv("SLURM_TASKS_PER_NODE");
    if (tasks_per_node) {
        return std::string(tasks_per_node);
    } else {
        return "no_slurm_tasks_per_node";
    }
}

// Let's get how many ranks are used from environment variable
std::string get_slurm_ntasks() {
    const char* ntasks = std::getenv("SLURM_NTASKS");
    if (ntasks) {
        return std::string(ntasks);
    } else {
        return "no_slurm_ntasks";
    }
}

// Let's get the rank of this process from environment variable
std::string get_slurm_rank() {
    const char* rank = std::getenv("SLURM_PROCID");
    if (rank) {
        return std::string(rank);
    } else {
        return "no_slurm_rank";
    }
}

// Let's get the node name from environment variable
std::string get_slurm_nodename() {
    const char* nodename = std::getenv("SLURM_NODELIST");
    if (nodename) {
        return std::string(nodename);
    } else {
        return "no_slurm_nodename";
    }
}

// Let's get the partition name from environment variable
std::string get_slurm_partition() {
    const char* partition = std::getenv("SLURM_JOB_PARTITION");
    if (partition) {
        return std::string(partition);
    } else {
        return "no_slurm_partition";
    }
}

std::string get_number_of_nodes() {
    const char* nodes = std::getenv("SLURM_JOB_NUM_NODES");
    if (nodes) {
        return std::string(nodes);
    } else {
        return "no_slurm_job_num_nodes";
    }
}

// All logs should be in the custom_logs directory
std::string custom_log_directory = "custom_logs/";

std::string get_log_file_name(std::string file_extension = ".log") {
    return get_number_of_nodes() + "x" +
           get_tasks_per_node() + "=" +
           get_slurm_ntasks() + "_" +
           get_slurm_partition() + "_" +
           get_slurm_nodename() + "_" +
           get_slurm_job_id() +
           file_extension;
}

template <size_t N>
void log_init(const std::string& message, const std::array<std::string, N>& context_parameter_names, const std::array<int, N>& context_parameter_values) {

    if (get_slurm_rank() != "0") {
        // Only rank 0 logs for now
        return;
    }

    step_times.clear();
    step_times.push_back(std::chrono::steady_clock::now().time_since_epoch().count());

    std::cout << "DEBUG TEST MSG 2: " << message << std::endl;

    // Check if directory exists, if not create it
    if (!std::filesystem::exists(custom_log_directory)) {
        std::filesystem::create_directory(custom_log_directory);
    }

    if (context_parameter_names.size() != context_parameter_values.size()) {
        // Handle error: sizes do not match
        std::cout << "Error: Context parameter names and values size mismatch." << std::endl;
        std::ofstream error_file(custom_log_directory + "error.txt");
        if (error_file.is_open()){
            error_file << "Error: Context parameter names and values size mismatch." << std::endl << "Names size: " << context_parameter_names.size() << ", Values size: " << context_parameter_values.size() << std::endl;
            error_file.close();
        }
        return;
    }

    std::string log_file_name = get_log_file_name();

    std::cout << "DEBUG TEST MSG 3: Log file path: " << custom_log_directory + log_file_name << std::endl;

    std::string log_file_path = custom_log_directory + log_file_name;

    std::ofstream log_file(log_file_path);
    if (log_file.is_open()) {
        log_file << "This is a test log file for SLURM job " << get_slurm_job_id() <<  ": " << message << std::endl;
        for (size_t i = 0; i < context_parameter_names.size(); ++i) {
            log_file << context_parameter_names[i] << ": " << context_parameter_values[i] << std::endl;
        }
        log_file << "----------------------------------------" << std::endl;
        log_file.close();
    }
}

void log_message(const std::string& message) {
    if (get_slurm_rank() != "0") {
        // Only rank 0 logs for now
        return;
    }

    std::string log_file_name = get_log_file_name();
    std::string log_file_path = custom_log_directory + log_file_name;

    std::ofstream log_file(log_file_path, std::ios_base::app); // Append mode
    if (log_file.is_open()) {
        log_file << message << std::endl;
        log_file.close();
    }
}

struct Step {
    int logicalTimeStep;
    int globalTimeStep;
    int restartTimeStep;
    bool isCouplingStep;
    bool isInferenceStep;
    int count = 1;
    // also save the time used for this step
    long long timestamp;
    long long steady_clock;

    Step(int logicalTimeStep_, int globalTimeStep_, int restartTimeStep_, bool isCouplingStep_, bool isInferenceStep_)
        : logicalTimeStep(logicalTimeStep_), globalTimeStep(globalTimeStep_), restartTimeStep(restartTimeStep_),
          isCouplingStep(isCouplingStep_), isInferenceStep(isInferenceStep_) {
            step_times.push_back(std::chrono::steady_clock::now().time_since_epoch().count());
        steady_clock = std::chrono::steady_clock::now().time_since_epoch().count() - step_times[step_times.size() - 2];
        timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
          }

    void update() {
        count++;
        step_times[step_times.size() - 1] = std::chrono::steady_clock::now().time_since_epoch().count();
        steady_clock = step_times[step_times.size() - 1] - step_times[step_times.size() - 2];
        timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    }
};

std::vector<Step> steps_log;

void log_step(int logicalTimeStep, int globalTimeStep, int restartTimeStep, bool isCouplingStep, bool isInferenceStep) {
    if (get_slurm_rank() != "0") {
        // Only rank 0 logs for now
        return;
    }

    if (!steps_log.empty() &&
        steps_log.back().logicalTimeStep == logicalTimeStep &&
        steps_log.back().globalTimeStep == globalTimeStep &&
        steps_log.back().restartTimeStep == restartTimeStep &&
        steps_log.back().isCouplingStep == isCouplingStep &&
        steps_log.back().isInferenceStep == isInferenceStep) {
        steps_log.back().update();
    } else {
        Step step = {logicalTimeStep, globalTimeStep, restartTimeStep, isCouplingStep, isInferenceStep};
        steps_log.push_back(step);
    }

}

void flush_steps_log() {
    if (get_slurm_rank() != "0") {
        // Only rank 0 logs for now
        return;
    }

    std::string log_file_name = get_log_file_name(".csv");
    std::string log_file_path = custom_log_directory + log_file_name;

    // check if file exists
    bool file_exists = std::filesystem::exists(log_file_path);

    std::ofstream log_file(log_file_path);
    if (log_file.is_open()) {
        
        log_file << "logical_time_step,global_time_step,restart_time_step,coupling,inference,count,timestamp,clock" << std::endl;
        for (const auto& step : steps_log) {
            log_file << step.logicalTimeStep << "," << step.globalTimeStep << "," << step.restartTimeStep << "," << step.isCouplingStep << "," << step.isInferenceStep << "," << step.count << "," << step.timestamp << "," << step.steady_clock << std::endl;
        }
        log_file.close();
    }
}

void log_deinit() {
    if (get_slurm_rank() != "0") {
        // Only rank 0 logs for now
        return;
    }

    std::cout << "DEBUG TEST MSG 4: Deinitializing log for SLURM job " << get_slurm_job_id() << std::endl;

    std::string log_file_name = get_log_file_name();
    std::string log_file_path = custom_log_directory + log_file_name;

    std::ofstream log_file(log_file_path, std::ios_base::app); // Append mode
    if (log_file.is_open()) {
        log_file << "End of log for SLURM job " << get_slurm_job_id() << std::endl;
        log_file.close();
    }

    flush_steps_log();
}