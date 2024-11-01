#gcc -o fft_example main.c -I/usr/include/mkl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl -liomp5
gcc -o fft_example fft_example.c -I/usr/include/mkl -lmkl_rt -lm