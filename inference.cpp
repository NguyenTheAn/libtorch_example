#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <model/CNN.hpp>
#include <utils/utils.hpp>

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
    Resnet18 model(2);
    torch::load(model, "../checkpoint.pt");

    cv::Mat img = cv::imread("../image.jpeg");
    img = crop_center(img);
    cv::resize(img, img, cv::Size(224,224));
    img.convertTo( img, CV_32FC3, 1/255.0 );
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    auto outputs = model->forward(img_tensor.to(device));
    auto pred = torch::argmax(outputs, 1).item<int>();;
    cv::putText(img, pred == 0 ? "cat" : "dog", cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 255, 0), 1);
    cv::imshow("image", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return(0);
}