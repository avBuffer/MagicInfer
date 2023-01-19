#ifndef MAGIC_INFER_DEMOS_IMAGE_UTIL_HPP_
#define MAGIC_INFER_DEMOS_IMAGE_UTIL_HPP_

#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


struct Detection 
{
    Rect box;
    float conf = 0.f;
    int class_id = -1;
};


float Letterbox(const Mat &image, Mat &out_image, const Size &new_shape = Size(640, 640), int stride = 32,
    const Scalar &color = Scalar(114, 114, 114), bool fixed_shape = false, bool scale_up = false);

void ScaleCoords(const Size &img_shape, Rect &coords, const Size &img_origin_shape);

#endif //MAGIC_INFER_DEMOS_IMAGE_UTIL_HPP_
