#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== AIx LibTorch Config ===" << std::endl;
    std::cout << torch::show_config() << std::endl;
    std::cout << "=== C++ ABI ===" << std::endl;
#if defined(_GLIBCXX_USE_CXX11_ABI)
    std::cout << "_GLIBCXX_USE_CXX11_ABI = " << _GLIBCXX_USE_CXX11_ABI << std::endl;
#else
    std::cout << "_GLIBCXX_USE_CXX11_ABI is NOT defined" << std::endl;
#endif
    return 0;
}
