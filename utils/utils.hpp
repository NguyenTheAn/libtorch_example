#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

int64_t str2int(std::string s){
    int64_t out = 0;
    for (auto c : s){
        int a = c-'0';
        out = out * 10 + a;
    }
    return out;
}

cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}