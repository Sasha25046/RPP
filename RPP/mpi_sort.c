#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0;
int ProcRank = 0;

void RandomDataInitialization(int* pArray, int Size) {
    srand((unsigned int)time(NULL) + ProcRank);
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
    if (m <= 0 || m >= n) return;
    int* tmp = (int*)malloc(n * sizeof(int));
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

void SaveArrayToFile(const int* pResult, int Size, const char* filename) {
    if (ProcRank != 0) return;

    printf("-> Trying to write to file '%s'...\n", filename);
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("!!! ERROR: Cannot open file %s for writing!\n", filename);
        return;
    }
    fprintf(f, "Size = %d\n", Size);
    for (int i = 0; i < Size; i++)
        fprintf(f, "%d\n", pResult[i]);
    fclose(f);
    printf("-> SUCCESS: File '%s' saved (Size: %d).\n", filename, Size);
}

void ProcessInitialization(int** ppArray, int** ppProcData, int** ppResult,
    int* pSize, int* pRowNum, int** ppSendCounts, int** ppDispls) {

    setvbuf(stdout, NULL, _IONBF, 0);

    if (ProcRank == 0) {
        do {
            printf("\nEnter size of the initial objects (array size): ");
            if (scanf("%d", pSize) != 1) {
                printf("Invalid input!\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (*pSize < ProcNum)
                printf("Size of the objects must be greater than number of processes!\n");
        } while (*pSize < ProcNum);
    }

    MPI_Bcast(pSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    *ppSendCounts = (int*)malloc(ProcNum * sizeof(int));
    *ppDispls = (int*)malloc(ProcNum * sizeof(int));

    int RestRows = *pSize;
    (*ppSendCounts)[0] = RestRows / ProcNum;
    (*ppDispls)[0] = 0;
    for (int i = 1; i < ProcNum; i++) {
        RestRows -= (*ppSendCounts)[i - 1];
        (*ppSendCounts)[i] = RestRows / (ProcNum - i);
        (*ppDispls)[i] = (*ppDispls)[i - 1] + (*ppSendCounts)[i - 1];
    }

    *pRowNum = (*ppSendCounts)[ProcRank];
    *ppProcData = (int*)malloc(*pRowNum * sizeof(int));

    if (ProcRank == 0) {
        *ppArray = (int*)malloc(*pSize * sizeof(int));
        *ppResult = (int*)malloc(*pSize * sizeof(int));
        RandomDataInitialization(*ppArray, *pSize);

        SaveArrayToFile(*ppArray, *pSize, "initial_data_parallel.txt");
    }
}

void DataDistribution(int* pArray, int* pProcData, int* pSendCounts, int* pDispls, int RowNum) {
    MPI_Scatterv(pArray, pSendCounts, pDispls, MPI_INT, pProcData, RowNum, MPI_INT, 0, MPI_COMM_WORLD);
}

void TestDistribution(int* pArray, int* pProcData, int Size, int RowNum) {
    if (ProcRank == 0) {
        printf("Initial Array:\n");
        PrintArray(pArray, Size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf("\nProcRank = %d, Local Size = %d\n", ProcRank, RowNum);
            PrintArray(pProcData, RowNum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void ParallelMergePhase(int** ppProcData, int* pRowNum) {
    int step = 1;
    while (step < ProcNum) {
        if (ProcRank % (2 * step) == 0) {
            int partner = ProcRank + step;
            if (partner < ProcNum) {
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int* recv_buf = (int*)malloc(recv_size * sizeof(int));
                MPI_Recv(recv_buf, recv_size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int new_size = *pRowNum + recv_size;
                int* merged = (int*)malloc(new_size * sizeof(int));
                memcpy(merged, *ppProcData, *pRowNum * sizeof(int));
                memcpy(merged + *pRowNum, recv_buf, recv_size * sizeof(int));

                Merge(merged, new_size, *pRowNum);

                free(*ppProcData);
                free(recv_buf);
                *ppProcData = merged;
                *pRowNum = new_size;
            }
        }
        else {
            int target = ProcRank - step;
            MPI_Send(pRowNum, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            MPI_Send(*ppProcData, *pRowNum, MPI_INT, target, 1, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }
}

void ResultCollection(int* pProcData, int* pResult, int Size) {
    if (ProcRank == 0) {
        memcpy(pResult, pProcData, Size * sizeof(int));
    }
}

void TestResult(int* pArray, int* pResult, int Size) {
    if (ProcRank != 0) return;

    int* pSerial = (int*)malloc(Size * sizeof(int));
    memcpy(pSerial, pArray, Size * sizeof(int));
    MergeSortSerial(pSerial, Size);

    int equal = 1;
    for (int i = 0; i < Size; i++) {
        if (pResult[i] != pSerial[i]) { equal = 0; break; }
    }

    if (equal)
        printf("\nThe results of serial and parallel algorithms are identical.\n");
    else
        printf("\nERROR: The results of serial and parallel algorithms are NOT identical!\n");

    free(pSerial);
}

void ProcessTermination(int* pArray, int* pProcData, int* pResult, int* pSendCounts, int* pDispls) {
    if (ProcRank == 0) {
        free(pArray);
        free(pResult);
    }
    free(pProcData);
    free(pSendCounts);
    free(pDispls);
}

int main(int argc, char* argv[]) {
    int* pArray = NULL;
    int* pProcData = NULL;
    int* pResult = NULL;
    int* pSendCounts = NULL;
    int* pDispls = NULL;
    int   Size = 0;
    int   RowNum = 0;
    double Start, Finish, Duration;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        printf("Parallel MPI Merge Sort program\n");
        printf("Number of available processes = %d\n", ProcNum);
    }

    ProcessInitialization(&pArray, &pProcData, &pResult, &Size, &RowNum, &pSendCounts, &pDispls);

    if (Size <= 20) {
        TestDistribution(pArray, pProcData, Size, RowNum);
    }

    Start = MPI_Wtime();

    DataDistribution(pArray, pProcData, pSendCounts, pDispls, RowNum);

    MergeSortSerial(pProcData, RowNum);

    ParallelMergePhase(&pProcData, &RowNum);

    ResultCollection(pProcData, pResult, Size);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    if (ProcRank == 0) {
        if (Size <= 20) {
            printf("\n Result Array:\n");
            PrintArray(pResult, Size);
        }
        printf("\n Time of execution: %f sec\n", Duration);

        SaveArrayToFile(pResult, Size, "sorted_result_parallel.txt");
    }

    TestResult(pArray, pResult, Size);

    ProcessTermination(pArray, pProcData, pResult, pSendCounts, pDispls);

    MPI_Finalize();
    return 0;
}