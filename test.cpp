#include <torch/script.h>
#include <torch/torch.h>

torch::Device get_device(){
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
    }
    return device;
}
torch::Device device = get_device();

int main(void) {
    torch::jit::script::Module module;
    module = torch::jit::load("../yolov7.torchscript.pt");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 640, 640}));
    auto output = module.forward(inputs);
    std::cout << output << '\n';
    return(0);
}