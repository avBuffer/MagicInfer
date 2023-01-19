#include "image_util.hpp"


float Letterbox(const Mat &image, Mat &out_image, const Size &new_shape, int stride,
    const Scalar &color, bool fixed_shape, bool scale_up) 
{
    Size shape = image.size();
    float r = min((float) new_shape.height / (float) shape.height, (float) new_shape.width / (float) shape.width);
    if (!scale_up) {
        r = min(r, 1.0f);
    }

    int new_unpad[2]{ (int) round((float) shape.width * r), (int) round((float) shape.height * r) };
    Mat tmp;
    if (shape.width != new_unpad[0] || shape.height != new_unpad[1]) {
        resize(image, tmp, Size(new_unpad[0], new_unpad[1]));
    } else {
        tmp = image.clone();
    }

    float dw = new_shape.width - new_unpad[0];
    float dh = new_shape.height - new_unpad[1];

    if (!fixed_shape) {
        dw = (float) ((int) dw % stride);
        dh = (float) ((int) dh % stride);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(round(dh - 0.1f));
    int bottom = int(round(dh + 0.1f));
    int left = int(round(dw - 0.1f));
    int right = int(round(dw + 0.1f));
    
    copyMakeBorder(tmp, out_image, top, bottom, left, right, BORDER_CONSTANT, color);
    return 1.0f / r;
}


template<typename T>
T clip(const T &n, const T &lower, const T &upper) { return max(lower, min(n, upper)); }


void ScaleCoords(const Size &img_shape, Rect &coords, const Size &img_origin_shape) 
{
    float gain = min((float) img_shape.height / (float) img_origin_shape.height,
                          (float) img_shape.width  / (float) img_origin_shape.width);

    int pad[2] = {(int) (((float) img_shape.width  - (float) img_origin_shape.width * gain)  / 2.0f),
                  (int) (((float) img_shape.height - (float) img_origin_shape.height * gain) / 2.0f)};

    coords.x = (int) round(((float) (coords.x - pad[0]) / gain));
    coords.y = (int) round(((float) (coords.y - pad[1]) / gain));

    coords.width  = (int) round(((float) coords.width  / gain));
    coords.height = (int) round(((float) coords.height / gain));

    coords.x = clip(coords.x, 0, img_origin_shape.width);
    coords.y = clip(coords.y, 0, img_origin_shape.height);

    coords.width  = clip(coords.width,  0, img_origin_shape.width);
    coords.height = clip(coords.height, 0, img_origin_shape.height);
}
