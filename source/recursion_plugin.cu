// File: recursion_plugin.cu
// Compile with, for example:
//    nvcc -shared -Xcompiler "-fPIC" -lcublas -lcusolver -o librecursion_plugin.so recursion_plugin.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

constexpr int BS = 18;  // Block size: 18x18 matrices

//--------------------------------------------------------------------------
// The Recursion structure holds all the data.  
// Note: For arrays, we assume they are stored contiguously in column–major order.
//  - psi_b, hpsi, pmn_b: each has dimension [BS x BS x kk] (total size: kk*BS*BS)
//  - hall: dimension [BS x BS x nhall x nmax]
//  - ee:   dimension [BS x BS x nee x ntype]
//  - lsham: dimension [BS x BS x ntype]
//  - atemp_b, b2temp_b: dimension [BS x BS x lld]
//  - izero: length kk;  iz: length kk (lookup table: atom type for each atom)
//--------------------------------------------------------------------------
struct Recursion {
    // Basic dimensions (all in C-style 0-indexed)
    int kk;    // total number of atoms
    int nmax;  // number of impurity atoms
    int lld;   // number of recursion steps

    // Extra dimensions for Hamiltonian arrays
    int nhall; // number of blocks in third dimension of hall (nhall = max(nn(:,1))+1)
    int nee;   // number of blocks in third dimension of ee (nee = max(nn(:,1))+1)
    int ntype; // number of unique atom types

    // Device arrays (all stored contiguously in column-major order)
    cuDoubleComplex* psi_b;    // [BS x BS x kk]
    cuDoubleComplex* hpsi;     // [BS x BS x kk]
    cuDoubleComplex* pmn_b;    // [BS x BS x kk]
    cuDoubleComplex* atemp_b;  // [BS x BS x lld]
    cuDoubleComplex* b2temp_b; // [BS x BS x lld]

    // Hamiltonian arrays
    cuDoubleComplex* hall;     // [BS x BS x nhall x nmax]
    cuDoubleComplex* ee;       // [BS x BS x nee x ntype]
    cuDoubleComplex* lsham;    // [BS x BS x ntype]

    // Host arrays (kept on CPU)
    int* izero;   // length: kk
    int* irlist;  // length: kk
    int irnum;    // number of active atoms

    // Lookup array for atom type (for each atom); length: kk
    int* iz;

    // cuBLAS and cuSOLVER handles
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
};

//--------------------------------------------------------------------------
// Kernel: compute locham for an impurity atom.
// For impurity atom with index i (0-indexed), we compute:
//    locham = hall(:,:,1,i) + lsham(:,:,iz)
// Here, hall is stored as a 4D array with dimensions: [BS x BS x nhall x nmax],
// and we want the block corresponding to third index 0 (i.e. Fortran index 1).
// lsham is stored as [BS x BS x ntype] and iz (atom type) is passed in (assumed 0-indexed).
//--------------------------------------------------------------------------
__global__
void computeLochamImpurity(const cuDoubleComplex* hall, const cuDoubleComplex* lsham,
                             cuDoubleComplex* locham,
                             int i, int nhall, int BS, int atom_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BS * BS) {
        // Compute hall index for impurity atom i, with third index = 0.
        // Layout: hall is [BS x BS x nhall x nmax] in column-major order.
        // The fastest varying index is the first (row), then second (column),
        // then third, then fourth.
        // So, the offset for hall(:,:,0,i) is:
        //    hall_offset = i * (nhall * BS * BS) + 0 * (BS * BS)
        int hall_index = i * (nhall * BS * BS) + idx;  // idx within block [0, BS*BS)
        // lsham: for atom type, stored as [BS x BS x ntype].
        int lsham_index = atom_type * (BS * BS) + idx;
        locham[idx] = cuCadd(hall[hall_index], lsham[lsham_index]);
    }
}

//--------------------------------------------------------------------------
// Kernel: compute locham for a bulk atom.
// For bulk atom with index i (0-indexed, i >= nmax), we use the ee array.
// We compute:
//    locham = ee(:,:,1, atom_type) + lsham(:,:,atom_type)
// Here, ee is stored as [BS x BS x nee x ntype], and we select third index 0.
//--------------------------------------------------------------------------
__global__
void computeLochamBulk(const cuDoubleComplex* ee, const cuDoubleComplex* lsham,
                         cuDoubleComplex* locham,
                         int atom_type, int nee, int BS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BS * BS) {
        int ee_index = atom_type * (nee * BS * BS) + idx;  // ee(:,:,0,atom_type)
        int lsham_index = atom_type * (BS * BS) + idx;
        locham[idx] = cuCadd(ee[ee_index], lsham[lsham_index]);
    }
}

//--------------------------------------------------------------------------
// Helper functions for batched GEMM remain unchanged.
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

void batchedGemm(cublasHandle_t handle,
                 cublasOperation_t transA, cublasOperation_t transB,
                 int m, int n, int k,
                 const cuDoubleComplex* alpha,
                 const std::vector<const cuDoubleComplex*>& A_array, int lda,
                 const std::vector<const cuDoubleComplex*>& B_array, int ldb,
                 const cuDoubleComplex* beta,
                 std::vector<cuDoubleComplex*>& C_array,
                 int batchCount)
{
    cublasStatus_t stat = cublasZgemmBatched(handle, transA, transB,
                           m, n, k,
                           alpha,
                           reinterpret_cast<const cuDoubleComplex* const*>(A_array.data()), lda,
                           reinterpret_cast<const cuDoubleComplex* const*>(B_array.data()), ldb,
                           beta,
                           reinterpret_cast<cuDoubleComplex**>(C_array.data()),
                           batchCount);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasZgemmBatched failed" << std::endl;
        exit(EXIT_FAILURE);
    }
}

//--------------------------------------------------------------------------
// The main recursion routine.
// This routine mimics the Fortran hop_b routine by computing hpsi for all atoms.
// For impurity atoms (i from 0 to nmax-1), it computes:
//    locham = hall(:,:,1,i) + lsham(:,:,iz[i])
// and for bulk atoms (i from nmax to kk-1):
//    locham = ee(:,:,1,iz[i]) + lsham(:,:,iz[i])
// Then, for each atom, it performs:
//    hpsi(:,:,i) = GEMM( locham, psi_b(:,:,i) )
// After that, it performs an accumulation step (as in your original code)
// and an eigen–decomposition (the “crecal” part) to update psi_b.
// (For brevity, the accumulation and eigen–decomposition parts are kept similar.)
//
// Note: Here we assume that the host lookup array iz (of length kk) has been
// copied into rec->iz, with atom types stored as 0-indexed integers.
//--------------------------------------------------------------------------
void runRecursion(Recursion* rec)
{
    const int matrixSize = BS;
    int threads = 256, blocks;

    // 1. Zero out hpsi on the device.
    size_t totalHpsiBytes = rec->kk * BS * BS * sizeof(cuDoubleComplex);
    cudaMemset(rec->hpsi, 0, totalHpsiBytes);

    // 2. Process impurity atoms: i = 0 ... nmax-1.
    for (int i = 0; i < rec->nmax; i++) {
        if (rec->izero[i] != 0) {
            // Read atom type from host lookup array rec->iz.
            // (For simplicity, we copy rec->iz to host for this atom.)
            int atom_type;
            cudaMemcpy(&atom_type, rec->iz + i, sizeof(int), cudaMemcpyDeviceToHost);
            // Launch kernel to compute locham for impurity atom i.
            cuDoubleComplex* d_locham;
            cudaMalloc(&d_locham, BS * BS * sizeof(cuDoubleComplex));
            blocks = (BS * BS + threads - 1) / threads;
            computeLochamImpurity<<<blocks, threads>>>(rec->hall, rec->lsham, d_locham, i, rec->nhall, BS, atom_type);
            cudaDeviceSynchronize();
            // GEMM: hpsi(:,:,i) = d_locham * psi_b(:,:,i)
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
            cublasStatus_t stat = cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              BS, BS, BS,
                                              &alpha,
                                              d_locham, BS,
                                              rec->psi_b + i * BS * BS, BS,
                                              &beta,
                                              rec->hpsi + i * BS * BS, BS);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasZgemm failed for impurity atom " << i << std::endl;
                exit(EXIT_FAILURE);
            }
            cudaFree(d_locham);
        }
    }

    // 3. Process bulk atoms: i = nmax ... kk-1.
    for (int i = rec->nmax; i < rec->kk; i++) {
        if (rec->izero[i] != 0) {
            int atom_type;
            cudaMemcpy(&atom_type, rec->iz + i, sizeof(int), cudaMemcpyDeviceToHost);
            cuDoubleComplex* d_locham;
            cudaMalloc(&d_locham, BS * BS * sizeof(cuDoubleComplex));
            blocks = (BS * BS + threads - 1) / threads;
            computeLochamBulk<<<blocks, threads>>>(rec->ee, rec->lsham, d_locham, atom_type, rec->nee, BS);
            cudaDeviceSynchronize();
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
            cublasStatus_t stat = cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                              BS, BS, BS,
                                              &alpha,
                                              d_locham, BS,
                                              rec->psi_b + i * BS * BS, BS,
                                              &beta,
                                              rec->hpsi + i * BS * BS, BS);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasZgemm failed for bulk atom " << i << std::endl;
                exit(EXIT_FAILURE);
            }
            cudaFree(d_locham);
        }
    }

    // 4. Build the active-atom list on the host.
    rec->irnum = 0;
    for (int i = 0; i < rec->kk; i++) {
        if (rec->izero[i] != 0) {
            rec->irlist[rec->irnum++] = i;
        }
    }

    // 5. Accumulate a summation matrix over the active atoms.
    //     For each active atom, compute: temp = (psi_b(:,:,atom))^H * hpsi(:,:,atom)
    // and then sum these to obtain "summ".
    std::vector<cuDoubleComplex> summ(BS * BS, make_cuDoubleComplex(0.0, 0.0));
    std::vector<cuDoubleComplex> tempMatrix(BS * BS);
    // We use a batched GEMM here.
    std::vector<const cuDoubleComplex*> psiB_active;
    std::vector<const cuDoubleComplex*> hpsi_active;
    for (int i = 0; i < rec->irnum; i++) {
        int idx = rec->irlist[i];
        psiB_active.push_back(rec->psi_b + idx * BS * BS);
        hpsi_active.push_back(rec->hpsi + idx * BS * BS);
    }
    // Allocate temporary device memory for each active atom’s result
    // and perform GEMM: temp_i = (psi_b)^H * hpsi.
    std::vector<cuDoubleComplex*> temp_ptrs = buildPointerArray(nullptr, rec->irnum);
    // For simplicity, we allocate one contiguous block for the batch.
    cuDoubleComplex* d_temp;
    cudaMalloc(&d_temp, rec->irnum * BS * BS * sizeof(cuDoubleComplex));
    temp_ptrs = buildPointerArray(d_temp, rec->irnum);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    batchedGemm(rec->cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N,
                BS, BS, BS,
                &alpha,
                psiB_active, BS,
                hpsi_active, BS,
                &beta,
                temp_ptrs, BS,
                rec->irnum);
    // Copy each 18x18 result back to host and sum them.
    for (int i = 0; i < rec->irnum; i++) {
         cudaMemcpy(tempMatrix.data(), d_temp + i * BS * BS,
                    BS * BS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
         for (int j = 0; j < BS * BS; j++) {
             summ[j] = cuCadd(summ[j], tempMatrix[j]);
         }
    }
    cudaFree(d_temp);
    // Save the summation matrix into atemp_b for the current recursion step.
    // For simplicity, we store it in atemp_b block 0.
    cudaMemcpy(rec->atemp_b, summ.data(), BS * BS * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // 6. CRECAL PART: Perform eigen–decomposition on the summation matrix in atemp_b[0],
    //    then form b = U * lam * U^H and b_inv = U * lam_inv * U^H,
    //    and update psi_b for the active atoms.
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
       // Compute b = U * lam, then b = (U * lam) * U^H.
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_U, matrixSize,
                   d_lam, matrixSize,
                   &beta,
                   d_b, matrixSize);
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_b, matrixSize,
                   d_U, matrixSize,
                   &beta,
                   d_b, matrixSize);
       // Compute b_inv similarly.
       cublasZgemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   d_U, matrixSize,
                   d_lam_inv, matrixSize,
                   &beta,
                   d_b_inv, matrixSize);
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

    // 7. Finally, update psi_b for the active atoms using b_inv.
    {
       std::vector<const cuDoubleComplex*> pmn_active_const = buildConstPointerArray(rec->pmn_b, rec->irnum);
       std::vector<cuDoubleComplex*> psiB_update = buildPointerArray(rec->psi_b, rec->irnum);
       // Broadcast d_b_inv for all active atoms.
       std::vector<const cuDoubleComplex*> b_inv_bcast(rec->irnum, d_b_inv);
       alpha = make_cuDoubleComplex(1.0, 0.0);
       beta  = make_cuDoubleComplex(0.0, 0.0);
       batchedGemm(rec->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   BS, BS, BS,
                   &alpha,
                   pmn_active_const, BS,
                   b_inv_bcast, BS,
                   &beta,
                   psiB_update, rec->irnum);
    }
    cudaFree(d_b);  cudaFree(d_b_inv);
    // End of recursion step.
}

//--------------------------------------------------------------------------
// Extern "C" interfaces to be called from Fortran.
// Note: The names here have trailing underscores to match gfortran's naming.
//--------------------------------------------------------------------------
extern "C" {

  // Create and initialize a Recursion structure.
  // The extra parameters: nhall, nee, ntype.
  Recursion* create_recursion_(int kk, int nmax, int lld, int nhall, int nee, int ntype)
  {
      Recursion* rec = new Recursion;
      rec->kk = kk;
      rec->nmax = nmax;
      rec->lld = lld;
      rec->nhall = nhall;
      rec->nee = nee;
      rec->ntype = ntype;
      rec->irnum = 0;
      size_t atomMatrixBytes = kk * BS * BS * sizeof(cuDoubleComplex);
      cudaMalloc(&rec->psi_b, atomMatrixBytes);
      cudaMalloc(&rec->hpsi,  atomMatrixBytes);
      cudaMalloc(&rec->pmn_b, atomMatrixBytes);
      cudaMalloc(&rec->atemp_b, lld * BS * BS * sizeof(cuDoubleComplex));
      cudaMalloc(&rec->b2temp_b, lld * BS * BS * sizeof(cuDoubleComplex));
      size_t hallBytes = nmax * nhall * BS * BS * sizeof(cuDoubleComplex);
      cudaMalloc(&rec->hall, hallBytes);
      size_t eeBytes = ntype * nee * BS * BS * sizeof(cuDoubleComplex);
      cudaMalloc(&rec->ee, eeBytes);
      size_t lshamBytes = ntype * BS * BS * sizeof(cuDoubleComplex);
      cudaMalloc(&rec->lsham, lshamBytes);
      rec->izero = new int[kk];
      rec->irlist = new int[kk];
      cudaMalloc(&rec->iz, kk * sizeof(int)); // For atom type lookup
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
          cudaFree(rec->iz);
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

  // Copy hall (size: nmax * nhall * BS * BS).
  void copy_hall_to_device_(Recursion* rec, const cuDoubleComplex* host_hall)
  {
      size_t bytes = rec->nmax * rec->nhall * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->hall, host_hall, bytes, cudaMemcpyHostToDevice);
  }

  // Copy ee (size: ntype * nee * BS * BS).
  void copy_ee_to_device_(Recursion* rec, const cuDoubleComplex* host_ee)
  {
      size_t bytes = rec->ntype * rec->nee * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->ee, host_ee, bytes, cudaMemcpyHostToDevice);
  }

  // Copy lsham (size: ntype * BS * BS).
  void copy_lsham_to_device_(Recursion* rec, const cuDoubleComplex* host_lsham)
  {
      size_t bytes = rec->ntype * BS * BS * sizeof(cuDoubleComplex);
      cudaMemcpy(rec->lsham, host_lsham, bytes, cudaMemcpyHostToDevice);
  }

  // Copy the Fortran integer array izero (length: kk) into rec->izero.
  void copy_izero_to_host_(Recursion* rec, const int* host_izero)
  {
      size_t bytes = rec->kk * sizeof(int);
      memcpy(rec->izero, host_izero, bytes);
  }

  // Copy the Fortran integer array iz (length: kk) into rec->iz.
  void copy_iz_to_device_(Recursion* rec, const int* host_iz)
  {
      size_t bytes = rec->kk * sizeof(int);
      cudaMemcpy(rec->iz, host_iz, bytes, cudaMemcpyHostToDevice);
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
