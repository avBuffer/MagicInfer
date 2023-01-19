#ifndef MAGIC_TICK_HPP_
#define MAGIC_TICK_HPP_

#include <iostream>
#include <chrono>

using namespace std;


#ifndef __ycm__

#define TICK(x) auto bench_##x = chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfs\n", #x, chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - bench_##x).count());

#else
#define TICK(x)
#define TOCK(x)
#endif

#endif //MAGIC_TICK_HPP_
