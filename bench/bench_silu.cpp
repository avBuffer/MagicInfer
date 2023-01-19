#include <benchmark/benchmark.h>
#include "data/tensor.hpp"

#if __SSE2__
#include <emmintrin.h>
#include "../include/utils/sse_math.hpp"
#endif

using namespace magic_infer;


static void BM_SiLuArma(benchmark::State &state) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(16, 320, 320);
    input->Rand();

    for (auto _ : state) {
        shared_ptr<Tensor<float>> output = input->Clone();
        output->Transform([](const float value) {
            return value / (1.f + exp(-value));
        });
    }
}


static void BM_SiLuSimd(benchmark::State &state) 
{
    __m128 _one = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(16, 320, 320);
    input->Rand();

    for (auto _ : state) {
        shared_ptr<Tensor<float>> output = input->Clone();
        uint32_t size = output->size();
        float *ptr = const_cast<float *>(output->RawPtr());

        for (uint32_t j = 0; j + 3 < size; j += 4) {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
    }
}


static void BM_SiLuSimdOMP(benchmark::State &state) 
{
    __m128 _one = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    
     shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(16, 320, 320);
    input->Rand();

    for (auto _ : state) {
        shared_ptr<Tensor<float>> output = input->Clone();
        uint32_t size = output->size();
        float *ptr = const_cast<float *>(output->RawPtr());

        int thread_count = 4;
#pragma omp parallel for num_threads(thread_count)
        for (uint32_t j = 0; j < size - 3; j += 4) {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_store_ps(ptr, _p);
#pragma omp critical
            ptr += 4;
        }
    }
}


BENCHMARK(BM_SiLuArma);
BENCHMARK(BM_SiLuSimd);
BENCHMARK(BM_SiLuSimdOMP);
