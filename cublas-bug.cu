#include <time.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "cublas_v2.h"


/*
 * Host code
 */
int main(int argc, char *argv[])
{
    // Compute C = transpose(A)*B for large A (>2^31 elements) using cuBLAS

    cublasHandle_t cublas_h;

    printf("Init cuBLAS...\n");
    cublasCreate(&cublas_h);

    float* A;
    float* B;
    float* C;

    // Size of main matrix A
    size_t Ar = 24000; // rows of A
    size_t Ac = 100000; // columns of A

    size_t Bc = 3; // Columns of B

    size_t N_A = Ar*Ac; // bug is triggered if Bc > 1 and N_A > 2^31
    size_t N_B = Ar*Bc;
    size_t N_C = Ac*Bc;

    cudaMallocManaged(&A, sizeof(float)*N_A);
    cudaMallocManaged(&B, sizeof(float)*N_B);
    cudaMallocManaged(&C, sizeof(float)*N_C);

    printf("N_A = %zi (%g)\n", N_A, (double)N_A);

    // Set values in A and B
    printf("Setting values for A and B\n");
    for(size_t i = 0; i < N_A; i++)
	A[i] = 1.0f;

    for(size_t i = 0; i < N_B; i++)
	B[i] = 1.0f;

    // Call cuBLAS - alpha*op(A)*op(B) + beta*C
    const size_t m = Ac;  // rows of op(A) and C
    const size_t n = Bc;   // columns of op(B) and C
    const size_t k = Ar; // columns op op(A) and rows of op(B)
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t lda = Ar;
    size_t ldb = Ar;
    size_t ldc = Ac;

    printf("Call cuBLAS\n");
    cublasStatus_t errn =
	cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
		&alpha, A, lda, B, ldb, &beta, C, ldc);

    if(errn == CUBLAS_STATUS_SUCCESS)
	printf("cuBLAS success\n");
    else
	printf("cuBLAS error\n");

    printf("cudaDeviceSynchronize()\n");
    cudaDeviceSynchronize();

    printf("C[0] = %g\n", C[0]); // This is correct == Ar
    printf("C[%zi] = %g\n", N_C-1, C[N_C-1]); // This is incorrect when Bc > 1 and N_A > 2^31

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
