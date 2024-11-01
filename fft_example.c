#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
//#include <complex.h>
#include <math.h>

void print_complex_array(const MKL_Complex8* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("(%f, %f)\n", arr[i].real, arr[i].imag);
    }
}

int main() {
    int n = 8;  // Taille de la FFT
    MKL_Complex8* x = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8));

    // Remplir le tableau avec des données d'exemple
    for (int i = 0; i < n; i++) {
        //x[i] = (MKL_Complex8)(ccos(2 * M_PI * i / n));
        x[i].real = cos(2 * M_PI * i / n); // Partie réelle
        x[i].imag = 0.0f;     // Partie imaginaire
    }

    print_complex_array(x, n);

    // Calcul de la FFT
    MKL_Complex8* y = (MKL_Complex8*)malloc(n * sizeof(MKL_Complex8));
    DFTI_DESCRIPTOR_HANDLE handle;
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, n);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(handle);
    DftiComputeForward(handle, x, y);
    DftiFreeDescriptor(&handle);

    // Afficher le résultat
    printf("Résultat de la FFT :\n");
    print_complex_array(y, n);

    // Libérer la mémoire
    free(x);
    free(y);

    return 0;
}
