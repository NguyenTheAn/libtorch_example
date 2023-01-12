#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <model/CNN.hpp>
#include <dataset/dogncat.hpp>

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
  Resnet18 model(2);
  model->to(device);

  int batch_size = 32;
  int epochs = 5;
  auto train = Dogncat("../dataset/dogncat/train/").map(torch::data::transforms::Stack<>());
  int train_size = train.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                      std::move(train), 
                      batch_size);
  auto val = Dogncat("../dataset/dogncat/valid/").map(torch::data::transforms::Stack<>());
  int val_size = val.size().value();
  auto val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                      std::move(val), 
                      batch_size);

  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));
  // double avgloss = 0;
  for (int e=1; e<=epochs; e++){
    double eloss = 0;
    int num = 0;
    model->train();
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto data = batch.data.to(torch::kFloat32);
      auto labels = batch.target.to(device);
      torch::Tensor outputs = model->forward(data.to(device));
      outputs = torch::softmax(outputs, 1);
      auto loss = torch::cross_entropy_loss(outputs, labels);
      loss.backward();
      optimizer.step();
      float l = loss.template item<float>();
      eloss += l;
      num++;
      // std::cout<<l<<std::endl;
      // break;
    }
    eloss /= num;
    float acc;
    int true_label = 0;
    model->eval();
    for (auto& batch : *val_loader){
      torch::NoGradGuard no_grad;
      auto data = batch.data.to(torch::kFloat32);
      auto labels = batch.target.to(device);
      torch::Tensor outputs = model->forward(data.to(device));
      outputs = torch::softmax(outputs, 1);
      auto res = torch::argmax(outputs, 1);
      auto labels_digit = torch::argmax(labels, 1);
      for (int i=0; i<res.sizes()[0]; i++){
        int a = res[i].item<int>();
        int b = labels_digit[i].item<int>();
        if (a == b) true_label += 1;
      }
    }
    acc = true_label * 1.0 / val_size;

    std::cout<<"Epoch "<<e<<": Loss "<<eloss<<" || Acc "<<acc<<std::endl;
  }
  torch::save(model, "../checkpoint.pt");
  // std::cout<<acc<<std::endl;

  return 0;
}