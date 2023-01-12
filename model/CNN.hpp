#include <torch/torch.h>

torch::nn::Conv2d conv3x3(int inC, int outC, int stride=1, int groups=1, int dilation=1){
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(inC, outC, 3).stride(stride).groups(groups).dilation(dilation).bias(false).padding(dilation));
}

torch::nn::Conv2d conv1x1(int inC, int outC, int stride=1){
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(inC, outC, 1).stride(stride).bias(false));
}

struct BasicBlockImpl : public torch::nn::Module{
    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr;
    torch::nn::ReLU relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::BatchNorm2d bn2 = nullptr;
    torch::nn::Sequential downsample = nullptr;
    static const int expansion = 1;
    BasicBlockImpl(int inC, int outC, torch::nn::Sequential, int);
    torch::Tensor forward(torch::Tensor x);
};

BasicBlockImpl::BasicBlockImpl(int inC, int outC, torch::nn::Sequential downsample = nullptr, int stride = 1){
    conv1 = register_module("conv1", conv3x3(inC, outC, stride));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(outC));
    conv2 = register_module("conv2", conv3x3(outC, outC));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(outC));
    register_module("relu", relu);
    if (downsample){
        this->downsample = downsample;
        register_module("downsample", this->downsample);
    }
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor input){
    auto identity = input.clone();
    auto x = this->conv1(input);
    x = this->bn1(x);
    x = this->relu(x);
    
    x = this->conv2(x);
    x = this->bn2(x);
    if (downsample){
        identity = downsample->forward(input);
    }
    x += identity;
    x = this->relu(x);
    
    return x;
}
TORCH_MODULE(BasicBlock);

struct Resnet18Impl : public torch::nn::Module{
    int inplanes = 64;
    torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, inplanes, 7).stride(2).bias(false).padding(3));
    torch::nn::BatchNorm2d bn1 = torch::nn::BatchNorm2d(inplanes);
    torch::nn::ReLU relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    torch::nn::MaxPool2d maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding(0));
    torch::nn::Sequential layer1 = nullptr;
    torch::nn::Sequential layer2 = nullptr;
    torch::nn::Sequential layer3 = nullptr;
    torch::nn::Sequential layer4 = nullptr;
    torch::nn::AdaptiveAvgPool2d avgpool = nullptr;
    torch::nn::Linear fc = nullptr;


    Resnet18Impl(int);
    torch::Tensor forward(torch::Tensor);
    torch::nn::Sequential make_layer(int, int, int);
};

torch::nn::Sequential Resnet18Impl::make_layer(int  planes, int numl, int stride = 1){
    torch::nn::Sequential downsample;
    bool use_downsample = false;
    if (stride != 1 || inplanes != planes * BasicBlock::Impl::expansion){
        use_downsample = true;
        downsample = torch::nn::Sequential(
            conv1x1(inplanes, planes * BasicBlock::Impl::expansion, stride),
            torch::nn::BatchNorm2d(planes)
        );
    }
    torch::nn::Sequential layer;
    if (use_downsample)
        layer->push_back(BasicBlock(inplanes, planes, downsample, stride));
    else layer->push_back(BasicBlock(inplanes, planes, nullptr, stride));
    inplanes = planes * BasicBlock::Impl::expansion;
    for (int i=1; i<numl; i++){
        layer->push_back(BasicBlock(inplanes, planes));
    }
    return layer;
}

Resnet18Impl::Resnet18Impl(int num_classes){
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu", relu);
    register_module("maxpool", maxpool);
    layer1 = register_module("layer1", make_layer(64, 2));
    layer2 = register_module("layer2", make_layer(128, 2, 2));
    layer3 = register_module("layer3", make_layer(256, 2, 2));
    layer4 = register_module("layer4", make_layer(512, 2, 2));
    avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1})));
    fc = register_module("fc", torch::nn::Linear(torch::nn::LinearOptions(512 * BasicBlock::Impl::expansion, num_classes)));
    
}

torch::Tensor Resnet18Impl::forward(torch::Tensor x){
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avgpool->forward(x);
    x = torch::flatten(x, 1);
    x = fc->forward(x);

    return x;
}

TORCH_MODULE(Resnet18);