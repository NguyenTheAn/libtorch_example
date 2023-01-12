#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <model/FCN.hpp>
#include <dataset/mnist.hpp>

torch::Device get_device(){
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
    }
    return device;
}
torch::Device device = get_device();

int main() {
  Net model;
  model.to(device);

  int batch_size = 32;
  int epochs = 5;
  auto mnist_train = Mnist("../dataset/mnist/train.csv").map(torch::data::transforms::Stack<>());
  int train_size = mnist_train.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                      std::move(mnist_train), 
                      batch_size);
  auto mnist_val = Mnist("../dataset/mnist/val.csv").map(torch::data::transforms::Stack<>());
  int val_size = mnist_val.size().value();
  auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                      std::move(mnist_val), 
                      batch_size);

  torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-4));

  for (int e=1; e<=epochs; e++){
    model.train();
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto data = batch.data.to(torch::kFloat32);
      auto labels = batch.target.to(device);
      torch::Tensor outputs = model.forward(data.to(device));
      auto loss = torch::cross_entropy_loss(outputs, labels);
      loss.backward();
      optimizer.step();
      float l = loss.template item<float>();

      std::cout<<l<<std::endl;
      // std::cout<<labels<<std::endl;
      // break;
    }
  }

  float acc;
  int true_label = 0;
  model.eval();
  for (auto& batch : *val_loader){
    torch::NoGradGuard no_grad;
    auto data = batch.data.to(torch::kFloat32);
    auto labels = batch.target.to(device);
    torch::Tensor outputs = model.forward(data.to(device));
    auto res = torch::argmax(outputs, 1);
    auto labels_digit = torch::argmax(labels, 1);
    for (int i=0; i<res.sizes()[0]; i++){
      int a = res[i].item<int>();
      int b = labels_digit[i].item<int>();
      if (a == b) true_label += 1;
    }
  }
  acc = true_label * 1.0 / val_size;
  std::cout<<acc<<std::endl;

  return 0;
}