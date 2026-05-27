#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void RandomDataInitialization(int* pArray, int Size) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < Size; i++)
        pArray[i] = rand() % 100000;
}

void PrintArray(const int* pArray, int Size) {
    int show = (Size > 20) ? 10 : Size;
    printf("  [");
    for (int i = 0; i < show; i++)
        printf("%d%s", pArray[i], (i < show - 1) ? ", " : "");
    if (Size > 20) {
        printf(", ..., ");
        for (int i = Size - 10; i < Size; i++)
            printf("%d%s", pArray[i], (i < Size - 1) ? ", " : "");
    }
    printf("]\n");
}

void Merge(int* a, int n, int m) {
    int* tmp = (int*)malloc(n * sizeof(int));
    if (!tmp) {
        printf("Error: Memory allocation failed in Merge.\n");
        return;
    }
    int i = 0, j = m, k = 0;
    while (i < m && j < n)
        tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < m) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, n * sizeof(int));
    free(tmp);
}

void MergeSortSerial(int* a, int n) {
    if (n < 2) return;
    int mid = n / 2;
    MergeSortSerial(a, mid);
    MergeSortSerial(a + mid, n - mid);
    Merge(a, n, mid);
}

#define OMP_THRESHOLD 2000

void MergeSortOMP(int* a, int n) {
    if (n < 2) return;
    if (n < OMP_THRESHOLD) {
        MergeSortSerial(a, n);
        return;
    }
    int mid = n / 2;

#pragma omp task shared(a) firstprivate(mid)
    MergeSortOMP(a, mid);

#pragma omp task shared(a) firstprivate(mid, n)
    MergeSortOMP(a + mid, n - mid);

#pragma omp taskwait
    Merge(a, n, mid);
}

int CheckResult(const int* serial_result, const int* omp_result, int Size) {
    for (int i = 1; i < Size; i++) {
        if (serial_result[i] < serial_result[i - 1]) return 0;
        if (omp_result[i] < omp_result[i - 1]) return 0;
    }
    for (int i = 0; i < Size; i++) {
        if (serial_result[i] != omp_result[i]) return 0;
    }
    return 1;
}

void SaveArrayToFile(const int* pArray, int Size, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("  Cannot open %s for writing.\n", filename);
        return;
    }
    fprintf(f, "Size = %d\n", Size);
    for (int i = 0; i < Size; i++)
        fprintf(f, "%d\n", pArray[i]);
    fclose(f);
    printf("  Data saved to '%s'\n", filename);
}

void ProcessInitialization(int** ppSerial, int** ppOMP, int* pSize, int* pNumThreads) {
    do {
        printf("\nEnter array size: ");
        if (scanf("%d", pSize) != 1) {
            printf("  Invalid input!\n");
            exit(1);
        }
        if (*pSize <= 0) printf("  Array size must be greater than 0!\n");
    } while (*pSize <= 0);

    int maxHardwareThreads = omp_get_max_threads();

    do {
        printf("Enter number of OpenMP threads (cores) [1-%d]: ", maxHardwareThreads);
        if (scanf("%d", pNumThreads) != 1) {
            printf("  Invalid input!\n");
            exit(1);
        }
        if (*pNumThreads <= 0 || *pNumThreads > maxHardwareThreads) {
            printf("  Number of threads must be between 1 and %d!\n", maxHardwareThreads);
        }
    } while (*pNumThreads <= 0 || *pNumThreads > maxHardwareThreads);

    printf("\nChosen size = %d, threads = %d\n", *pSize, *pNumThreads);

    *ppSerial = (int*)malloc(*pSize * sizeof(int));
    *ppOMP = (int*)malloc(*pSize * sizeof(int));

    if (!*ppSerial || !*ppOMP) {
        printf("Error: Memory allocation failed for arrays.\n");
        exit(1);
    }

    RandomDataInitialization(*ppSerial, *pSize);
    SaveArrayToFile(*ppSerial, *pSize, "initial_data_omp.txt");
    memcpy(*ppOMP, *ppSerial, *pSize * sizeof(int));
}

void ProcessTermination(int* pSerial, int* pOMP) {
    free(pSerial);
    free(pOMP);
}

int main() {
    int* pSerial = NULL;
    int* pOMP = NULL;
    int   Size = 0;
    int   NumThreads = 0;
    double t1, t2;

    printf("=== Serial & Parallel OpenMP Merge Sort program ===\n");

    ProcessInitialization(&pSerial, &pOMP, &Size, &NumThreads);


    t1 = omp_get_wtime();
    MergeSortSerial(pSerial, Size);
    t2 = omp_get_wtime();
    double serialTime = t2 - t1;
    printf("\nSerial Merge Sort time : %f sec\n", serialTime);

    omp_set_num_threads(NumThreads);

    t1 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        {
            MergeSortOMP(pOMP, Size);
        }
    }
    t2 = omp_get_wtime();
    double ompTime = t2 - t1;
    printf("OpenMP Merge Sort time : %f sec (threads = %d)\n", ompTime, NumThreads);


    printf("\nSpeedup (serial / OMP): %.2f\n", serialTime / ompTime);

    printf("\nVerifying results... ");
    if (CheckResult(pSerial, pOMP, Size)) {
        printf("The results of serial and OpenMP algorithms are identical.\n");
    }
    else {
        printf("ERROR: results mismatch or array not sorted!\n");
    }

    SaveArrayToFile(pSerial, Size, "sorted_result_serial_omp.txt");
    ProcessTermination(pSerial, pOMP);

    printf("\nPress Enter to exit...\n");
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
    getchar();
    return 0;
}