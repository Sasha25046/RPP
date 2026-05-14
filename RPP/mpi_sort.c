#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
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

int main(int argc, char** argv) {
    int rank, size, n = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setvbuf(stdout, NULL, _IONBF, 0);

    if (rank == 0) {
        printf("Enter array size: ");
        fflush(stdout);
        if (scanf("%d", &n) != 1) n = 0;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        MPI_Finalize();
        return 0;
    }

    int* data = NULL;
    if (rank == 0) {
        data = (int*)malloc(n * sizeof(int));
        srand((unsigned int)time(NULL));
        for (int i = 0; i < n; i++) data[i] = rand() % 10000;
    }

    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int base = n / size;
    int rem = n % size;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_n = sendcounts[rank];
    int* sub_array = (int*)malloc(local_n * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_mpi = MPI_Wtime();

    MPI_Scatterv(data, sendcounts, displs, MPI_INT, sub_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    merge_sort_serial(sub_array, local_n);

    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int* recv_data = (int*)malloc(recv_size * sizeof(int));
                MPI_Recv(recv_data, recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int* merged = (int*)malloc((local_n + recv_size) * sizeof(int));
                memcpy(merged, sub_array, local_n * sizeof(int));
                memcpy(merged + local_n, recv_data, recv_size * sizeof(int));
                merge(merged, local_n + recv_size, local_n);
                free(sub_array); free(recv_data);
                sub_array = merged;
                local_n += recv_size;
            }
        }
        else {
            int target = rank - step;
            MPI_Send(&local_n, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            MPI_Send(sub_array, local_n, MPI_INT, target, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (rank == 0) {
        printf("MPI (Tree Merge): %f sec\n", MPI_Wtime() - start_mpi);
        free(data);
    }

    free(sub_array); free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}