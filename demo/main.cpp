#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "data/tensor.hpp"
#include "image_util.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/tick.hpp"

using namespace std;
using namespace cv;
using namespace magic_infer;


map<int, string> load_name(const string &classes_file)
{
    assert(!classes_file.empty());

    map<int, string> class_names;
    ifstream in(classes_file);
    if (!in) {
        cout<<"failed to load classes file: " + classes_file + ".\n";
        exit(0);
    }

    string line;
    int index = 0;
    while (getline(in, line)) {
        class_names[index] = line;
        index += 1;
    }
    return class_names;
}


void SingleImageYoloInferNano(const string &param_file, const string &weight_file, const string &image_file, 
    const string &classes_file, const string &out_path, const float conf_thresh = 0.25f, const float iou_thresh = 0.25f) 
{
    assert(!image_file.empty());
    const int32_t input_c = 3;
    const int32_t input_h = 640;
    const int32_t input_w = 640;

    RuntimeGraph graph(param_file, weight_file);
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    map<int, string> class_names = load_name(classes_file);

    int repeat_times = 5;
    Mat image = imread(image_file);
    assert(!image.empty());
    const int32_t origin_input_h = image.size().height;
    const int32_t origin_input_w = image.size().width;

    int stride = 32;
    Mat out_image;
    Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114}, true);

    Mat rgb_image;
    cvtColor(out_image, rgb_image, COLOR_BGR2RGB);

    Mat normalize_image;
    rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

    vector<Mat> split_images;
    split(normalize_image, split_images);
    assert(split_images.size() == input_c);

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(input_c, input_h, input_w);
    input->Fill(0.f);

    int index = 0;
    int offset = 0;
    for (const auto &split_image : split_images) {
        assert(split_image.total() == input_w * input_h);
        const Mat &split_image_t = split_image.t();
        memcpy(input->at(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
        index += 1;
        offset += split_image.total();
    }
    
    int    font_face = FONT_HERSHEY_COMPLEX;
    double font_scale = 2;
    int    thickness = 2;

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    for (int rpt = 0; rpt < repeat_times; ++rpt) {
        TICK(FORWARD)
        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, true);
        TOCK(FORWARD);

        assert(outputs.size() == inputs.size());
        assert(outputs.size() == 1);
        const auto &output = outputs.front();
        const auto &shapes = output->shapes();
        assert(shapes.size() == 3);

        const uint32_t batch = shapes.at(0);
        assert(batch == 1);
        const uint32_t elements = shapes.at(1);
        const uint32_t num_info = shapes.at(2);
        vector<Detection> detections;

        vector<Rect> boxes;
        vector<float> confs;
        vector<int> class_ids;

        const uint32_t b = 0;
        for (uint32_t i = 0; i < elements; ++i) {
            float cls_conf = output->at(b, i, 4);
            if (cls_conf >= conf_thresh) {
                int center_x = (int) (output->at(b, i, 0));
                int center_y = (int) (output->at(b, i, 1));
                int width = (int) (output->at(b, i, 2));
                int height = (int) (output->at(b, i, 3));
                int left = center_x - width / 2;
                int top = center_y - height / 2;

                int best_class_id = -1;
                float best_conf = -1.f;
                for (uint32_t j = 5; j < num_info; ++j) {
                    if (output->at(b, i, j) > best_conf) {
                        best_conf = output->at(b, i, j);
                        best_class_id = int(j - 5);
                    }
                }

                boxes.emplace_back(left, top, width, height);
                confs.emplace_back(best_conf * cls_conf);
                class_ids.emplace_back(best_class_id);
            }
        }

        vector<int> indices;
        dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);

        for (int idx : indices) {
            Detection det;
            det.box = Rect(boxes[idx]);
            ScaleCoords(Size{input_w, input_h}, det.box, Size{origin_input_w, origin_input_h});

            det.conf = confs[idx];
            det.class_id = class_ids[idx];
            detections.emplace_back(det);
        }

        for (const auto &detection : detections) {
            rectangle(image, detection.box, Scalar(0, 255, 0), thickness);
            putText(image, class_names[detection.class_id], Point(detection.box.x, detection.box.y), 
                font_face, font_scale, Scalar(0, 255, 255), thickness);
        }

        // if (rpt == 0) {
        string out_file = out_path + "/output_" + to_string(rpt) + ".jpg";
        imwrite(out_file, image);
        cout << "SingleImageYoloInferNano reap_time=" << rpt << ", out_file=" << out_file << endl;
        // }
    }
}


void MultiImageYoloInferNano(const string &param_file, const string &weight_file, const string &image_file,
    const string &classes_file, const string &out_path, const float conf_thresh = 0.25f, const float iou_thresh = 0.25f) 
{
    const int32_t input_c = 3;
    const int32_t input_h = 640;
    const int32_t input_w = 640;

    RuntimeGraph graph(param_file, weight_file);
    graph.Build("pnnx_input_0", "pnnx_output_0");

    map<int, string> class_names = load_name(classes_file);

    const int batch_size = 8;
    vector<Mat> images;
    vector<shared_ptr<Tensor<float>>> inputs;

// #pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; ++i) {
        assert(!image_file.empty());
        Mat image = imread(image_file);
        assert(!image.empty());
        images.push_back(image);

        int stride = 32;
        Mat out_image;
        Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114}, true);

        Mat rgb_image;
        cvtColor(out_image, rgb_image, COLOR_BGR2RGB);

        Mat normalize_image;
        rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

        vector<Mat> split_images;
        split(normalize_image, split_images);
        assert(split_images.size() == input_c);

        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(input_c, input_h, input_w);
        input->Fill(0.f);

        int index = 0, offset = 0;
        for (const auto &split_image : split_images) {
            assert(split_image.total() == input_w * input_h);
            const Mat &split_image_t = split_image.t();
            memcpy(input->at(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
            index  += 1;
            offset += split_image.total();
        }
        
        inputs.push_back(input);
    }

    TICK(FORWARD)
    vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, true);
    TOCK(FORWARD);

    assert(outputs.size() == inputs.size());
    assert(outputs.size() == batch_size);

    int    font_face = FONT_HERSHEY_COMPLEX;
    double font_scale = 2;
	int    thickness = 2;

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &image = images.at(i);
        const int32_t origin_input_h = image.size().height;
        const int32_t origin_input_w = image.size().width;

        const auto &output = outputs.at(i);
        assert(!output->empty());
        const auto &shapes = output->shapes();
        assert(shapes.size() == 3);

        const uint32_t elements = shapes.at(1);
        const uint32_t num_info = shapes.at(2);
        vector<Detection> detections;

        vector<Rect> boxes;
        vector<float> confs;
        vector<int> class_ids;

        const uint32_t b = 0;
        for (uint32_t e = 0; e < elements; ++e) {
            float cls_conf = output->at(b, e, 4);
            if (cls_conf >= conf_thresh) {
                int center_x = (int) (output->at(b, e, 0));
                int center_y = (int) (output->at(b, e, 1));
                
                int width  = (int) (output->at(b, e, 2));
                int height = (int) (output->at(b, e, 3));

                int left   = center_x - width / 2;
                int top    = center_y - height / 2;

                int best_class_id = -1;
                float best_conf = -1.f;
                for (uint32_t j = 5; j < num_info; ++j) {
                    if (output->at(b, e, j) > best_conf) {
                        best_conf = output->at(b, e, j);
                        best_class_id = int(j - 5);
                    }
                }

                boxes.emplace_back(left, top, width, height);
                confs.emplace_back(best_conf * cls_conf);
                class_ids.emplace_back(best_class_id);
            }
        }

        vector<int> indices;
        dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);

        for (int idx : indices) {
            Detection det;
            det.box = Rect(boxes[idx]);
            ScaleCoords(Size{input_w, input_h}, det.box, Size{origin_input_w, origin_input_h});

            det.conf = confs[idx];
            det.class_id = class_ids[idx];
            detections.emplace_back(det);
        }

        for (const auto &detection : detections) {
            rectangle(image, detection.box, Scalar(0, 255, 0), thickness);
            putText(image, class_names[detection.class_id], Point(detection.box.x, detection.box.y), 
                font_face, font_scale, Scalar(0, 0, 255), thickness);
        }

        string out_file = out_path + "/output_multi_" + to_string(i) + ".jpg";
        imwrite(out_file, image);
        cout << "MultiImageYoloInferNano i=" << i << ", out_file=" << out_file << endl;
    }
}


int main() 
{
    const string &param_file  = "../../weights/yolo/demo/yolov5s_batch8.pnnx.param";
    const string &weight_file = "../../weights/yolo/demo/yolov5s_batch8.pnnx.bin";
    
    const string &image_file = "../../data/imgs/bus.jpg";
    const string &classes_file  = "../../data/classes/coco.names";
    const string &out_path  = "../../out/";

    //判断该文件夹是否存在
    if (access(out_path.c_str(), 0) == -1) {
        if (mkdir(out_path.c_str(), S_IRWXU) != 0) { //创建失败
            cout << "Fail to create directory : " << out_path << endl;
            throw exception();
        }
    }
    
    // SingleImageYoloInferNano(param_file, weight_file, image_file, classes_file, out_path);

    // for (int i = 0; i < 32; ++i) {
        MultiImageYoloInferNano(param_file, weight_file, image_file, classes_file, out_path);
    // }
    return 0;
}
