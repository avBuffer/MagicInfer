#include <benchmark/benchmark.h>
#include "data/tensor.hpp"

#if __SSE2__
#include <emmintrin.h>
#include "../include/utils/sse_math.hpp"
#endif

using namespace magic_infer;


static void BM_ReLuArma(benchmark::State &state) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(16, 320, 320);
    input->Rand();

    for (auto _ : state) {
        shared_ptr<Tensor<float>> output = input->Clone();
        output->Transform([](const float value) {
            return 1.f / (1.f + exp(-value));
        });
    }
}


static void BM_ReLuSimd(benchmark::State &state) 
{
    __m128 _one1 = _mm_set1_ps(1.f);
    __m128 _one2 = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(16, 320, 320);
    input->Rand();

    for (auto _ : state) {
        shared_ptr<Tensor<float>> output = input->Clone();
        uint32_t size = output->size();
        float *ptr = const_cast<float *>(output->RawPtr());

        for (uint32_t j = 0; j + 3 < size; j += 4) {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_div_ps(_one1, _mm_add_ps(_one2, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
    }
}


BENCHMARK(BM_ReLuArma);
BENCHMARK(BM_ReLuSimd);
