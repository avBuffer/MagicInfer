#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

using namespace magic_infer;


TEST(test_tensor, tensor_init1) 
{
    Tensor<float> f1(3, 224, 224);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
}


TEST(test_tensor, tensor_init2) 
{
    Tensor<float> f1(vector<uint32_t>{3, 224, 224});
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
}


TEST(test_tensor, set_data) 
{
    Tensor<float> f2(3, 224, 224);
    arma::fcube cube1(224, 224, 3);
    cube1.randn();
    f2.set_data(cube1);
    ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}


TEST(test_tensor, data) 
{
    Tensor<float> f2(3, 224, 224);
    f2.Fill(1.f);
    arma::fcube cube1(224, 224, 3);
    cube1.fill(1.);
    f2.set_data(cube1);
    ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}


TEST(test_tensor, empty) 
{
    Tensor<float> f2;
    ASSERT_EQ(f2.empty(), true);
    Tensor<float> f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
}


TEST(test_tensor, transform) 
{
    Tensor<float> f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    f3.Transform([](const float &value) { return 1.f; });
    for (int i = 0; i < f3.size(); ++i) {
        ASSERT_EQ(f3.index(i), 1.f);
    }
}


TEST(test_tensor, clone) 
{
    Tensor<float> f3(3, 3, 3);
    ASSERT_EQ(f3.empty(), false);
    f3.Rand();

    const auto &f4 = f3.Clone();
    ASSERT_EQ(f4->size(), f3.size());
    for (int i = 0; i < f3.size(); ++i) {
        ASSERT_EQ(f3.index(i), f4->index(i));
    }
}


TEST(test_tensor, flatten) 
{
    Tensor<float> f3(3, 3, 3);
    vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(float(i));
    }

    f3.Fill(values);
    f3.Flatten();
    ASSERT_EQ(f3.channels(), 1);
    ASSERT_EQ(f3.rows(), 27);
    ASSERT_EQ(f3.cols(), 1);
}


TEST(test_tensor, fill_at1) 
{
    Tensor<float> f3(3, 3, 3);
    vector<float> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(float(i));
    }

    f3.Fill(values);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < f3.rows(); ++i) {
            for (int j = 0; j < f3.cols(); ++j) {
                ASSERT_EQ(f3.at(c, i, j), index);
                index += 1;
            }
        }
    }
}


TEST(test_tensor, fill_at2)
{
    Tensor<float> f3(3, 3, 3);
    vector<float> values;
    f3.Fill(1.f);

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < f3.rows(); ++i) {
            for (int j = 0; j < f3.cols(); ++j) {
                ASSERT_EQ(f3.at(c, i, j), 1.f);
            }
        }
    }
}


TEST(test_tensor, add1) 
{
    const auto &f1 = make_shared<Tensor<float>>(3, 224, 224);
    f1->Fill(1.f);
    const auto &f2 = make_shared<Tensor<float>>(3, 224, 224);
    f2->Fill(2.f);
    const auto &f3 = Tensor<float>::ElementAdd(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 3.f);
    }
}


TEST(test_tensor, mul1) 
{
    const auto &f1 = make_shared<Tensor<float>>(3, 224, 224);
    f1->Fill(3.f);
    const auto &f2 = make_shared<Tensor<float>>(3, 224, 224);
    f2->Fill(2.f);
    const auto &f3 = Tensor<float>::ElementMultiply(f2, f1);
    for (int i = 0; i < f3->size(); ++i) {
        ASSERT_EQ(f3->index(i), 6.f);
    }
}


TEST(test_tensor, shapes) 
{
    Tensor<float> f3(2, 3, 4);
    const vector<uint32_t> shapes = f3.shapes();
    ASSERT_EQ(shapes.at(0), 2);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 4);
}


TEST(test_tensor, raw_shapes1) 
{
    Tensor<float> f3(2, 3, 4);
    f3.ReRawshape({24});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(shapes.at(0), 24);
}


TEST(test_tensor, raw_shapes2) 
{
    Tensor<float> f3(2, 3, 4);
    f3.ReRawshape({4, 6});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 2);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_shapes3) {
    Tensor<float> f3(2, 3, 4);
    f3.ReRawshape({4, 3, 2});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 3);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 2);
}


TEST(test_tensor, raw_view1) 
{
    Tensor<float> f3(2, 3, 4);
    f3.ReRawView({24});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 1);
    ASSERT_EQ(shapes.at(0), 24);
}


TEST(test_tensor, raw_view2) 
{
    Tensor<float> f3(2, 3, 4);
    f3.ReRawView({4, 6});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 2);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 6);
}


TEST(test_tensor, raw_view3) 
{
    Tensor<float> f3(2, 3, 4);
    f3.ReRawView({4, 3, 2});
    const auto &shapes = f3.raw_shapes();
    ASSERT_EQ(shapes.size(), 3);
    ASSERT_EQ(shapes.at(0), 4);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 2);
}


TEST(test_tensor, padding) 
{
    Tensor<float> tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.Fill(1.f);
    tensor.Padding({1, 1, 1, 1}, 0);
    ASSERT_EQ(tensor.rows(), 6);
    ASSERT_EQ(tensor.cols(), 7);

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int r = 0; r < tensor.rows(); ++r) {
            for (int c_ = 0; c_ < tensor.cols(); ++c_) {
                if (c_ == 0 || r == 0) {
                    ASSERT_EQ(tensor.at(c, r, c_), 0);
                }
                index += 1;
            }
        }
    }
}


TEST(test_tensor, review) 
{
    Tensor<float> tensor(3, 4, 5);
    vector<float> values;
    for (int i = 0; i < 60; ++i) {
        values.push_back(float(i));
    }

    tensor.Fill(values);
    tensor.ReRawView({4, 3, 5});
    
    const auto &data = tensor.at(0);
    int index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_EQ(data.at(i, j), index);
            index += 1;
        }
    }
}
