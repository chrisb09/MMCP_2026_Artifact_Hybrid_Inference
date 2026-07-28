#include <torch/script.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t kSamples = 3456;
constexpr int64_t kInputSequence = 5;
constexpr int64_t kFields = 3;
constexpr int64_t kCubeDimension = 8;
constexpr int64_t kForecastWindow = 2;
constexpr int64_t kFeaturesPerStep = 512;

std::vector<float> read_floats(const std::string& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input: " + path);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("input has an unexpected size: " + path);
    }
    return values;
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: direct_libtorch_inference MODEL INPUT OUTPUT BATCH_SIZE\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::string input_path = argv[2];
        const std::string output_path = argv[3];
        const int64_t batch_size = std::stoll(argv[4]);
        if (batch_size <= 0 || kSamples % batch_size != 0) {
            throw std::runtime_error("batch size must be a positive divisor of 3456");
        }

        auto input_values = read_floats(input_path, kSamples * kInputSequence * kFeaturesPerStep);
        auto model = torch::jit::load(model_path, torch::kCPU);
        model.eval();
        std::cout << torch::show_config() << '\n';

        const auto input = torch::from_blob(
            input_values.data(),
            {kSamples, kInputSequence, kFeaturesPerStep},
            torch::TensorOptions().dtype(torch::kFloat32));

        std::vector<torch::Tensor> outputs;
        outputs.reserve(kSamples / batch_size);
        torch::NoGradGuard no_grad;
        for (int64_t start = 0; start < kSamples; start += batch_size) {
            std::vector<torch::jit::IValue> inputs{input.slice(0, start, start + batch_size)};
            outputs.push_back(model.forward(inputs).toTensor().to(torch::kCPU));
        }

        const auto result = torch::cat(outputs, 0).contiguous();
        const std::vector<int64_t> expected_shape{
            kSamples, kForecastWindow, kFeaturesPerStep};
        if (result.sizes().vec() != expected_shape) {
            throw std::runtime_error("model returned an unexpected shape: " + result.toString());
        }

        std::ofstream output(output_path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot open output: " + output_path);
        output.write(static_cast<const char*>(result.data_ptr()),
                     static_cast<std::streamsize>(result.numel() * sizeof(float)));
        std::cout << "batch=" << batch_size << " output=" << output_path
                  << " min=" << result.min().item<float>()
                  << " max=" << result.max().item<float>() << '\n';
    } catch (const c10::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
