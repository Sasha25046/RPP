#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

void merge(int* a, int n, int m) {
    int i = 0, j = m, k = 0;
    int* x = (int*)malloc(n * sizeof(int));

    while (i < m && j < n) {
        x[k++] = (a[i] < a[j]) ? a[i++] : a[j++];
    }
    while (i < m) x[k++] = a[i++];
    while (j < n) x[k++] = a[j++];

    memcpy(a, x, n * sizeof(int));
    free(x);
}

void merge_sort_serial(int* a, int n) {
    if (n < 2) return;

    int mid = n / 2;
    merge_sort_serial(a, mid);
    merge_sort_serial(a + mid, n - mid);
    merge(a, n, mid);
}

void merge_sort_omp(int* a, int n) {
    if (n < 2) return;

    if (n < 1000) {
        merge_sort_serial(a, n);
        return;
    }

    int mid = n / 2;
#pragma omp task shared(a)
    merge_sort_omp(a, mid);

#pragma omp task shared(a)
    merge_sort_omp(a + mid, n - mid);

#pragma omp taskwait
    merge(a, n, mid);
}

int main() {
    int n = 0;

    printf("Enter array size: ");
    if (scanf("%d", &n) != 1 || n <= 0) return 0;

    int* data_serial = (int*)malloc(n * sizeof(int));
    int* data_omp = (int*)malloc(n * sizeof(int));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++) {
        int val = rand() % 10000;
        data_serial[i] = val;
        data_omp[i] = val;
    }

    double t1 = omp_get_wtime();
    merge_sort_serial(data_serial, n);
    printf("Serial: %f sec\n", omp_get_wtime() - t1);

    double t2 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        merge_sort_omp(data_omp, n);
    }
    printf("OpenMP: %f sec\n", omp_get_wtime() - t2);

    free(data_serial);
    free(data_omp);
    return 0;
}