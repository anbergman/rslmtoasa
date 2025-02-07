// File: recursion_plugin.cu
// Compile with, e.g.:
//    nvcc -shared -Xcompiler "-fPIC" -lcublas -lcusolver -o librecursion_plugin.so recursion_plugin.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>  // for memcpy

// Our “small” block size: 18x18 matrices.
constexpr int BS = 18;

//--------------------------------------------------------------------------
// The Recursion structure holds all the data (that in Fortran were in "this%...")
//--------------------------------------------------------------------------
struct Recursion {
    // Lattice/control parameters.
    int kk;    // total number of atoms
    int nmax;  // number of impurity atoms (first nmax are impurity)
    int lld;   // number of recursion steps

    // Device arrays for 18x18 blocks.
    cuDoubleComplex* psi_b;    // Working vector psi [kk][BS][BS]
    cuDoubleComplex* hpsi;     // H applied to psi [kk][BS][BS]
    cuDoubleComplex* pmn_b;    // Intermediate array [kk][BS][BS]
    cuDoubleComplex* atemp_b;  // Recursion history (or summation) [lld][BS][BS]
    cuDoubleComplex* b2temp_b; // Storage for previous B [lld][BS][BS]

    // Hamiltonian arrays (device memory):
    cuDoubleComplex* hall;  // Impurity Hamiltonian [nmax][BS][BS]
    cuDoubleComplex* ee;    // Bulk Hamiltonian [(kk-nmax)][BS][BS]
    cuDoubleComplex* lsham; // Local spin–orbit correction [kk][BS][BS]

    // Host arrays (allocated on the CPU)
    int* izero;   // Array of length kk (flags: nonzero = active)
    int* irlist;  // List of active atom indices (max length = kk)
    int irnum;    // Number of active atoms found

    // cuBLAS and cuSOLVER handles
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
};

//--------------------------------------------------------------------------
// Helper: launch a kernel that adds two BSxBS matrices (batched).
// Computes: C = A + B for batchCount matrices stored contiguously.
//--------------------------------------------------------------------------
__global__
void kernelBatchedMatrixAdd(const cuDoubleComplex* A,
                            const cuDoubleComplex* B,
                            cuDoubleComplex* C,
                            int batchCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchCount * BS * BS;
    if (idx < totalElements) {
        C[idx] = cuCadd(A[idx], B[idx]);
    }
}

//--------------------------------------------------------------------------
// Helper functions to build host pointer arrays for batched cuBLAS calls.
// Given a base pointer to contiguous device data, these functions return a
// vector of pointers (one per block).
//--------------------------------------------------------------------------
std::vector<const cuDoubleComplex*> buildConstPointerArray(const cuDoubleComplex* base, int count) {
    std::vector<const cuDoubleComplex*> ptrs(count);
    for (int i = 0; i < count; i++) {
        ptrs[i] = base + i * BS * BS;
    }
    return ptrs;
}

std::vector<cuDoubleComplex*> buildPointerArray(cuDoubleComplex* base, int count) {
    std::vector<cuDoubleComplex*> ptrs(count);
    for (int i = 0; i < count; i++) {
        ptrs[i] = base + i * BS * BS;
    }
    return ptrs;
}

//--------------------------------------------------------------------------
// Wrapper for batched GEMM using cuBLAS.
// Note: We use reinterpret_cast to convert the vector data pointers to the
// type expected by cublasZgemmBatched.
//--------------------------------------------------------------------------
void batchedGemm(cublasHandle_t handle,
                 cublasOperation_t transA, cublasOperation_t transB,
                 int m, int n, int k,
                 const cuDoubleComplex* alpha,
                 const std::vector<const cuDoubleComplex*>& A_array, int lda,
                 const std::vector<const cuDoubleComplex*>& B_array, int ldb,
                 const cuDoubleComplex* beta,
                 std::vector<cuDoubleComplex*>& C_array, int ldc,
                 int batchCount)
{
    cublasStatus_t stat = cublasZgemmBatched(handle, transA, transB,
                           m, n, k,
                           alpha,
                           reinterpret_cast<const cuDoubleComplex* const*>(A_array.data()), lda,
                           reinterpret_cast<const cuDoubleComplex* const*>(B_array.data()), ldb,
                           beta,
                           reinterpret_cast<cuDoubleComplex**>(C_array.data()), ldc,
                           batchCount);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasZgemmBatched failed" << std::endl;
        exit(EXIT_FAILURE);
    }
}

//--------------------------------------------------------------------------
// The main recursion routine (merging hop_b and crecal_b operations).
// (A simplified version; neighbor contributions and multi-step recursion
// are not shown in full detail.)
//--------------------------------------------------------------------------
void runRecursion(Recursion* rec)
{
    const int matrixSize = BS;
    int threads = 256, blocks;

    // 1. Zero out hpsi on the device.
    size_t totalHpsiBytes = rec->kk * BS * BS * sizeof(cuDoubleComplex);
    cudaMemset(rec->hpsi, 0, totalHpsiBytes);

    // --- HOP PART ---
    // Process impurity atoms (indices 0 .. nmax-1)
    int impurityCount = rec->nmax;
    cuDoubleComplex* d_locham_imp;
    cudaMalloc(&d_locham_imp, impurityCount * BS * BS * sizeof(cuDoubleComplex));
    int totalElemsImp = impurityCount * BS * BS;
    blocks = (totalElemsImp + threads - 1) / threads;
    kernelBatchedMatrixAdd<<<blocks, threads>>>(rec->hall, rec->lsham, d_locham_imp, impurityCount);
    cudaDeviceSynchronize();
    {
       std::vector<const cuDoubleComplex*> psiB_imp = buildConstPointerArray(rec->psi_b, impurityCount);
       std::vector<const cuDoubleComplex*> locham_imp = buildConstPointerArray(d_locham_imp, impurityCount);
       std::vector<cuDoubleComplex*> hpsi_imp = buildPointerArray(rec->hpsi, impurityCount);
       cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
       batchedGemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   locham_imp, matrixSize,
                   psiB_imp, matrixSize,
                   &beta,
                   hpsi_imp, matrixSize,
                   impurityCount);
    }
    cudaFree(d_locham_imp);

    // Process bulk atoms (indices nmax .. kk-1)
    int bulkCount = rec->kk - rec->nmax;
    cuDoubleComplex* d_locham_bulk;
    cudaMalloc(&d_locham_bulk, bulkCount * BS * BS * sizeof(cuDoubleComplex));
    int totalElemsBulk = bulkCount * BS * BS;
    blocks = (totalElemsBulk + threads - 1) / threads;
    kernelBatchedMatrixAdd<<<blocks, threads>>>(rec->ee + rec->nmax * BS * BS,
                                                 rec->lsham + rec->nmax * BS * BS,
                                                 d_locham_bulk, bulkCount);
    cudaDeviceSynchronize();
    {
       std::vector<const cuDoubleComplex*> psiB_bulk = buildConstPointerArray(rec->psi_b + rec->nmax * BS * BS, bulkCount);
       std::vector<const cuDoubleComplex*> locham_bulk = buildConstPointerArray(d_locham_bulk, bulkCount);
       std::vector<cuDoubleComplex*> hpsi_bulk = buildPointerArray(rec->hpsi + rec->nmax * BS * BS, bulkCount);
       cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
       batchedGemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   locham_bulk, matrixSize,
                   psiB_bulk, matrixSize,
                   &beta,
                   hpsi_bulk, matrixSize,
                   bulkCount);
    }
    cudaFree(d_locham_bulk);
    // --- END HOP PART ---

    // 2. Build the active-atom list (from the host copy of izero).
    rec->irnum = 0;
    for (int i = 0; i < rec->kk; i++) {
        if (rec->izero[i] != 0) {
            rec->irlist[rec->irnum++] = i;
        }
    }
    // Prepare pointer arrays for active atoms.
    std::vector<const cuDoubleComplex*> psiB_active(rec->irnum);
    std::vector<cuDoubleComplex*> hpsi_active(rec->irnum);
    std::vector<cuDoubleComplex*> pmn_active(rec->irnum);
    for (int i = 0; i < rec->irnum; i++) {
        int idx = rec->irlist[i];
        // Cast rec->psi_b to a const pointer for psiB_active.
        psiB_active[i] = static_cast<const cuDoubleComplex*>(rec->psi_b + idx * BS * BS);
        hpsi_active[i] = rec->hpsi + idx * BS * BS;
        pmn_active[i]   = rec->pmn_b + idx * BS * BS;
    }
    // Build a vector<const cuDoubleComplex*> from hpsi_active:
    std::vector<const cuDoubleComplex*> hpsi_active_const(rec->irnum);
    for (int i = 0; i < rec->irnum; i++) {
        hpsi_active_const[i] = hpsi_active[i];
    }
    // 2a. Update pmn_b = hpsi - pmn_b using a loop (since a batched GEAM is not available).
    {
       cuDoubleComplex alpha_geam = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta_geam  = make_cuDoubleComplex(-1.0, 0.0);
       for (int i = 0; i < rec->irnum; i++) {
         // The GEAM routine in some cuBLAS versions takes 12 arguments.
            cublasStatus_t stat = cublasZgeam(rec->cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            BS, BS,
                            &alpha_geam,
                            hpsi_active[i], BS,
                            &beta_geam,
                            pmn_active[i], BS,
                            pmn_active[i], BS);
         // cublasStatus_t stat = cublasZgeam(rec->cublasHandle,
         //                    CUBLAS_OP_N, CUBLAS_OP_N,
         //                    BS, BS,
         //                    &alpha_geam,
         //                    hpsi_active[i], BS,
         //                    &beta_geam,
         //                    pmn_active[i], BS,
         //                    pmn_active[i]);
         if (stat != CUBLAS_STATUS_SUCCESS) {
           std::cerr << "cublasZgeam failed in runRecursion" << std::endl;
           exit(EXIT_FAILURE);
         }
       }
    }
    // 3. Accumulate a summation matrix over active atoms:
    //     summ = sum_i ( psi_b[i]^H * hpsi[i] )
    cuDoubleComplex* d_temp;
    cudaMalloc(&d_temp, rec->irnum * BS * BS * sizeof(cuDoubleComplex));
    {
       cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
       // For the conjugate transpose on psiB_active, we use CUBLAS_OP_C.
       std::vector<const cuDoubleComplex*> psiB_active_conj = psiB_active;
       std::vector<cuDoubleComplex*> temp_ptrs = buildPointerArray(d_temp, rec->irnum);
       batchedGemm(rec->cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N,
                   BS, BS, BS,
                   &alpha,
                   psiB_active_conj, BS,
                   hpsi_active_const, BS,
                   &beta,
                   temp_ptrs, BS,
                   rec->irnum);
    }
    // Now copy each 18x18 result back to host and sum them.
    std::vector<cuDoubleComplex> summ(BS * BS, make_cuDoubleComplex(0.0, 0.0));
    std::vector<cuDoubleComplex> tempMatrix(BS * BS);
    if (d_temp == NULL) {
    printf("ERROR: d_temp is NULL before cudaMemcpy!\n");
    return;
}
cudaError_t err = cudaMalloc((void**)&d_temp, rec->irnum * BS * BS * sizeof(cuDoubleComplex));
if (err != cudaSuccess) {
    printf("ERROR: cudaMalloc for d_temp failed: %s\n", cudaGetErrorString(err));
    return;
}
int i=0 ;
if (i >= rec->irnum) {
    printf("ERROR: Out-of-bounds memory access! i = %d, rec->irnum = %d\n", i, rec->irnum);
    return;
}
printf("DEBUG: Copying from d_temp[%d] to host, size = %ld bytes\n",
       i * BS * BS, BS * BS * sizeof(cuDoubleComplex));

cudaMemcpy(tempMatrix.data(), d_temp + i * BS * BS,
           BS * BS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rec->irnum; i++) {
         cudaMemcpy(tempMatrix.data(), d_temp + i * BS * BS,
                    BS * BS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
         for (int j = 0; j < BS * BS; j++) {
             summ[j] = cuCadd(summ[j], tempMatrix[j]);
         }
    }
    cudaFree(d_temp);
    // Save the summation matrix into atemp_b[0] (for this example step).
    cudaMemcpy(rec->atemp_b, summ.data(), BS * BS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // --- CRECAL PART (Eigen–decomposition and update) ---
    int lwork = 0, info = 0;
    cusolverDnZheevd_bufferSize(rec->cusolverHandle, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_UPPER,
                                matrixSize, rec->atemp_b, matrixSize,
                                nullptr, &lwork);
    cuDoubleComplex* d_work;
    cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex));
    int* devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    std::vector<double> eigenvalues(matrixSize, 0.0);
    cusolverDnZheevd(rec->cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                  matrixSize, rec->atemp_b, matrixSize, eigenvalues.data(),
                  d_work, lwork, devInfo);
    // cusolverDnZheevd(rec->cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    //                   matrixSize, rec->atemp_b, matrixSize, eigenvalues.data(),
    //                   d_work, lwork, nullptr, 0, devInfo);
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0) {
        std::cerr << "Eigen decomposition failed with info = " << info << std::endl;
        exit(EXIT_FAILURE);
    }
    // Now rec->atemp_b holds the eigenvectors (U). Copy them to host.
    std::vector<cuDoubleComplex> U(matrixSize * matrixSize);
    cudaMemcpy(U.data(), rec->atemp_b, matrixSize * matrixSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    std::vector<cuDoubleComplex> lam(matrixSize * matrixSize, make_cuDoubleComplex(0.0, 0.0));
    std::vector<cuDoubleComplex> lam_inv(matrixSize * matrixSize, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < matrixSize; i++) {
         double sqrt_val = std::sqrt(eigenvalues[i]);
         lam[i * matrixSize + i] = make_cuDoubleComplex(sqrt_val, 0.0);
         lam_inv[i * matrixSize + i] = make_cuDoubleComplex(1.0 / sqrt_val, 0.0);
    }
    // Copy U, lam, and lam_inv to device.
    cuDoubleComplex* d_U;  cudaMalloc(&d_U, matrixSize * matrixSize * sizeof(cuDoubleComplex));
    cuDoubleComplex* d_lam;  cudaMalloc(&d_lam, matrixSize * matrixSize * sizeof(cuDoubleComplex));
    cuDoubleComplex* d_lam_inv;  cudaMalloc(&d_lam_inv, matrixSize * matrixSize * sizeof(cuDoubleComplex));
    cudaMemcpy(d_U, U.data(), matrixSize * matrixSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lam, lam.data(), matrixSize * matrixSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lam_inv, lam_inv.data(), matrixSize * matrixSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    // Form b = U * lam * U^H and b_inv = U * lam_inv * U^H.
    cuDoubleComplex* d_b;      cudaMalloc(&d_b, matrixSize * matrixSize * sizeof(cuDoubleComplex));
    cuDoubleComplex* d_b_inv;  cudaMalloc(&d_b_inv, matrixSize * matrixSize * sizeof(cuDoubleComplex));
    {
       cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
       // Use non-batched GEMM for single matrices.
       // First compute temp = U * lam.
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_U, matrixSize,
                   d_lam, matrixSize,
                   &beta,
                   d_b, matrixSize); // reuse d_b as temporary storage
       // Then b = temp * U^H.
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_b, matrixSize,
                   d_U, matrixSize,
                   &beta,
                   d_b, matrixSize);
       // Now compute b_inv = U * lam_inv * U^H.
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_U, matrixSize,
                   d_lam_inv, matrixSize,
                   &beta,
                   d_b_inv, matrixSize); // reuse d_b_inv as temporary
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_b_inv, matrixSize,
                   d_U, matrixSize,
                   &beta,
                   d_b_inv, matrixSize);
    }
    cudaFree(d_U);  cudaFree(d_lam);  cudaFree(d_lam_inv);
    cudaFree(devInfo);  cudaFree(d_work);

    // 4. Finally, update psi_b for the active atoms using the new b_inv.
    {
       std::vector<const cuDoubleComplex*> pmn_active_const = buildConstPointerArray(rec->pmn_b, rec->irnum);
       std::vector<cuDoubleComplex*> psiB_update = buildPointerArray(rec->psi_b, rec->irnum);
       // Broadcast d_b_inv for all atoms.
       std::vector<const cuDoubleComplex*> b_inv_bcast(rec->irnum, d_b_inv);
       cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
       cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
       batchedGemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   BS, BS, BS,
                   &alpha,
                   pmn_active_const, BS,
                   b_inv_bcast, BS,
                   &beta,
                   psiB_update, BS,
                   rec->irnum);
    }
    cudaFree(d_b);  cudaFree(d_b_inv);
    // End of the recursion step.
}

//--------------------------------------------------------------------------
// Extern "C" interfaces to be called from Fortran.
//--------------------------------------------------------------------------
extern "C" {

  // Create and initialize a Recursion structure.
  Recursion* create_recursion_(int kk, int nmax, int lld)
  {
      Recursion* rec = new Recursion;
      rec->kk = kk;
      rec->nmax = nmax;
      rec->lld = lld;
      rec->irnum = 0;
      size_t atomMatrixBytes = kk * BS * BS * sizeof(cuDoubleComplex);
      cudaMalloc(&rec->psi_b, atomMatrixBytes);
      cudaMalloc(&rec->hpsi,  atomMatrixBytes);
      cudaMalloc(&rec->pmn_b, atomMatrixBytes);
      cudaMalloc(&rec->hall, nmax * BS * BS * sizeof(cuDoubleComplex));
      cudaMalloc(&rec->ee,   (kk - nmax) * BS * BS * sizeof(cuDoubleComplex));
      cudaMalloc(&rec->lsham, atomMatrixBytes);
      cudaMalloc(&rec->atemp_b, lld * BS * BS * sizeof(cuDoubleComplex));
      cudaMalloc(&rec->b2temp_b, lld * BS * BS * sizeof(cuDoubleComplex));
      rec->izero = new int[kk];
      rec->irlist = new int[kk];
      cublasCreate(&rec->cublasHandle);
      cusolverDnCreate(&rec->cusolverHandle);
      return rec;
  }

  // Free all memory and destroy the Recursion structure.
  void destroy_recursion_(Recursion* rec)
  {
      if (rec) {
          cudaFree(rec->psi_b);
          cudaFree(rec->hpsi);
          cudaFree(rec->pmn_b);
          cudaFree(rec->hall);
          cudaFree(rec->ee);
          cudaFree(rec->lsham);
          cudaFree(rec->atemp_b);
          cudaFree(rec->b2temp_b);
          delete [] rec->izero;
          delete [] rec->irlist;
          cublasDestroy(rec->cublasHandle);
          cusolverDnDestroy(rec->cusolverHandle);
          delete rec;
      }
  }

  // Copy the Fortran psi_b array (host memory) into the device psi_b.
  void copy_psi_b_to_device_(Recursion* rec, const cuDoubleComplex* host_psi_b)
  {
      size_t bytes = rec->kk * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->psi_b, host_psi_b, bytes, cudaMemcpyHostToDevice);
  }

  // Copy hall (size: nmax*BS*BS).
  void copy_hall_to_device_(Recursion* rec, const cuDoubleComplex* host_hall)
  {
      size_t bytes = rec->nmax * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->hall, host_hall, bytes, cudaMemcpyHostToDevice);
  }

  // Copy ee (size: (kk - nmax)*BS*BS).
  void copy_ee_to_device_(Recursion* rec, const cuDoubleComplex* host_ee)
  {
      size_t bytes = (rec->kk - rec->nmax) * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->ee, host_ee, bytes, cudaMemcpyHostToDevice);
  }

  // Copy lsham (size: kk*BS*BS).
  void copy_lsham_to_device_(Recursion* rec, const cuDoubleComplex* host_lsham)
  {
      size_t bytes = rec->kk * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->lsham, host_lsham, bytes, cudaMemcpyHostToDevice);
  }

  // Copy the Fortran integer array izero (length: kk) into rec->izero.
  void copy_izero_to_host_(Recursion* rec, const int* host_izero)
  {
      size_t bytes = rec->kk * sizeof(int);
      memcpy(rec->izero, host_izero, bytes);
  }

  // Copy the device psi_b array to a Fortran host array.
  void copy_psi_b_to_host_(Recursion* rec, cuDoubleComplex* host_psi_b)
  {
      size_t bytes = rec->kk * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(host_psi_b, rec->psi_b, bytes, cudaMemcpyDeviceToHost);
  }

  // Copy the device atemp_b array to a Fortran host array.
  void copy_atemp_b_to_host_(Recursion* rec, cuDoubleComplex* host_atemp_b)
  {
      size_t bytes = rec->lld * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(host_atemp_b, rec->atemp_b, bytes, cudaMemcpyDeviceToHost);
  }

  // Copy the device b2temp_b array to a Fortran host array.
  void copy_b2temp_b_to_host_(Recursion* rec, cuDoubleComplex* host_b2temp_b)
  {
      size_t bytes = rec->lld * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(host_b2temp_b, rec->b2temp_b, bytes, cudaMemcpyDeviceToHost);
  }

  // Run one recursion step.
  void run_recursion_step_(Recursion* rec)
  {
      runRecursion(rec);
  }
}
