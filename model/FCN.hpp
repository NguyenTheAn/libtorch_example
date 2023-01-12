#include <torch/torch.h>

class Net : public torch::nn::Module{
    private:
        torch::nn::Linear linear1 = nullptr;
        torch::nn::Linear linear2 = nullptr;
        torch::nn::Linear linear3 = nullptr;
        torch::nn::Linear linear4 = nullptr;
        torch::nn::Linear linear5 = nullptr;
        torch::Tensor b1;
        torch::Tensor b2;
        torch::Tensor b3;
        torch::Tensor b4;
    public:

        Net();

        torch::Tensor forward(torch::Tensor input);
};

Net::Net(){
    linear1 = register_module("linear1", torch::nn::Linear(28*28, 512));
    b1 = register_parameter("b1", torch::randn(512));
    linear2 = register_module("linear2", torch::nn::Linear(512, 256));
    b2 = register_parameter("b2", torch::randn(256));
    linear3 = register_module("linear3", torch::nn::Linear(256, 128));
    b3 = register_parameter("b3", torch::randn(128));
    linear4 = register_module("linear4", torch::nn::Linear(128, 32));
    b4 = register_parameter("b4", torch::randn(32));
    linear5 = register_module("linear5", torch::nn::Linear(32, 10));
}

torch::Tensor Net::forward(torch::Tensor x){
    x = torch::relu(linear1(x) + b1);
    x = torch::relu(linear2(x) + b2);
    x = torch::relu(linear3(x) + b3);
    x = torch::relu(linear4(x) + b4);
    x = torch::softmax(linear5(x), 1);
    return x;
}