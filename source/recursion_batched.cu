// recursion_batched.cu
// Compile with: nvcc -lcublas -lcusolver recursion_batched.cu -o recursion_batched

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// For double–precision complex numbers.
#include <cuComplex.h>
using Complex = cuDoubleComplex;
#define MAKE_COMPLEX(a, b) make_cuDoubleComplex((a), (b))

// Our “small” block size (18×18 matrices)
constexpr int BS = 18;

// ----------------------------------------------------------------------
// Data structure for the recursion object.
// (In a real code, you’d likely fill in many more fields.)
//
// All 18x18 matrices are stored contiguously on the device. For example,
// psi_b, hpsi, and pmn_b are allocated as device arrays of size:
//      (number_of_atoms) * BS * BS
// Similarly, atemp_b (and b2temp_b) are allocated as device arrays of size:
//      (number_of_recursion_steps) * BS * BS
// ----------------------------------------------------------------------
struct Recursion {
    // Lattice info
    int kk;    // total number of atoms
    int nmax;  // number of impurity atoms (first nmax are impurity)
    int lld;   // number of recursion steps

    // Device arrays (each block is BS x BS)
    Complex* psi_b;    // working vector psi (size: kk * BS * BS)
    Complex* hpsi;     // H applied to psi (size: kk * BS * BS)
    Complex* pmn_b;    // intermediate array (size: kk * BS * BS)
    Complex* atemp_b;  // storage for summation (size: lld * BS * BS)
    Complex* b2temp_b; // storage for previous B (size: lld * BS * BS)

    // Hamiltonian arrays
    // For the impurity region we assume “hall” is stored for atoms 0...nmax-1.
    // For the bulk we assume “ee” is stored for atoms nmax...kk-1.
    Complex* hall;    // impurity Hamiltonian (size: nmax * BS * BS)
    Complex* ee;      // bulk Hamiltonian (size: (kk - nmax) * BS * BS)
    // lsham is assumed to be available for every atom (or atom type) as needed.
    // For simplicity we assume it is stored contiguously for kk atoms.
    Complex* lsham;   // local spin–orbit (size: kk * BS * BS)

    // Host arrays (not on device)
    int* izero;   // an array of length kk (flags; nonzero = active)
    int* irlist;  // list of active atom indices (size at least kk)
    int irnum;    // number of active atoms found
};

// ----------------------------------------------------------------------
// A simple kernel that adds two sets of BS×BS matrices stored contiguously.
// For batchCount matrices, each matrix is stored consecutively.
// C = A + B.
// ----------------------------------------------------------------------
__global__
void kernelBatchedMatrixAdd(const Complex* A, const Complex* B, Complex* C, int batchCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchCount * BS * BS;
    if (idx < totalElements) {
        C[idx] = cuCadd(A[idx], B[idx]);
    }
}

// ----------------------------------------------------------------------
// Helper functions to build host pointer arrays to the blocks
// in a contiguous device array.
// (For example, if psi_b is allocated as [kk][BS][BS], then the pointer
// for block i is: psi_b + i * BS * BS)
// ----------------------------------------------------------------------
std::vector<const Complex*> buildConstPointerArray(const Complex* base, int count) {
    std::vector<const Complex*> ptrs(count);
    for (int i = 0; i < count; i++) {
        ptrs[i] = base + i * BS * BS;
    }
    return ptrs;
}

std::vector<Complex*> buildPointerArray(Complex* base, int count) {
    std::vector<Complex*> ptrs(count);
    for (int i = 0; i < count; i++) {
        ptrs[i] = base + i * BS * BS;
    }
    return ptrs;
}

// ----------------------------------------------------------------------
// A wrapper to call the batched GEMM (cuBLAS version)
// This wraps cublasZgemmBatched.
// ----------------------------------------------------------------------
void batchedGemm(cublasHandle_t handle,
                 cublasOperation_t transA, cublasOperation_t transB,
                 int m, int n, int k,
                 const Complex* alpha,
                 const std::vector<const Complex*>& A_array, int lda,
                 const std::vector<const Complex*>& B_array, int ldb,
                 const Complex* beta,
                 std::vector<Complex*>& C_array, int ldc,
                 int batchCount)
{
    cublasStatus_t stat = cublasZgemmBatched(handle, transA, transB,
                           m, n, k,
                           alpha,
                           A_array.data(), lda,
                           B_array.data(), ldb,
                           beta,
                           C_array.data(), ldc, batchCount);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasZgemmBatched failed" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------
// Our main routine that merges the hop_b and crecal_b steps.
// In this simplified example we assume that (i) the impurity and bulk
// contributions are handled by nearly identical batched calls, (ii) neighbor
// contributions are omitted (or could be added in additional batched groups),
// and (iii) a lookup “dispatch” is performed by simply calling the proper
// batched routine.
// ----------------------------------------------------------------------
void runRecursion(Recursion &rec, cublasHandle_t handle, cusolverDnHandle_t cusolver)
{
    // (1) Zero out hpsi for all atoms.
    int totalHpsiBytes = rec.kk * BS * BS * sizeof(Complex);
    cudaMemset(rec.hpsi, 0, totalHpsiBytes);

    // --- HOP PART ---
    // Process impurity atoms (atoms 0 to nmax-1)
    int impurityCount = rec.nmax;
    // Allocate temporary device memory for the combined Hamiltonian: locham = hall + lsham.
    Complex* d_locham_imp;
    cudaMalloc(&d_locham_imp, impurityCount * BS * BS * sizeof(Complex));
    // For impurity atoms we assume that hall and lsham are stored consecutively.
    // Compute: d_locham_imp = hall + lsham (for atoms 0 .. impurityCount-1)
    int totalElemsImp = impurityCount * BS * BS;
    int threads = 256;
    int blocks = (totalElemsImp + threads - 1) / threads;
    kernelBatchedMatrixAdd<<<blocks, threads>>>(rec.hall, rec.lsham, d_locham_imp, impurityCount);
    cudaDeviceSynchronize();

    // Now update hpsi for impurity atoms:
    //   hpsi[i] = d_locham_imp[i] * psi_b[i]
    std::vector<const Complex*> psiB_imp = buildConstPointerArray(rec.psi_b, impurityCount);
    std::vector<const Complex*> locham_imp = buildConstPointerArray(d_locham_imp, impurityCount);
    std::vector<Complex*> hpsi_imp = buildPointerArray(rec.hpsi, impurityCount);
    Complex alpha = MAKE_COMPLEX(1.0, 0.0);
    Complex beta  = MAKE_COMPLEX(0.0, 0.0);
    batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                BS, BS, BS,
                &alpha,
                locham_imp, BS,
                psiB_imp, BS,
                &beta,
                hpsi_imp, BS,
                impurityCount);
    cudaFree(d_locham_imp);

    // Process bulk atoms (atoms nmax to kk-1)
    int bulkCount = rec.kk - rec.nmax;
    Complex* d_locham_bulk;
    cudaMalloc(&d_locham_bulk, bulkCount * BS * BS * sizeof(Complex));
    // For bulk atoms, assume ee and lsham are arranged so that the i‑th bulk atom
    // corresponds to index (nmax + i) in psi_b and hpsi.
    int totalElemsBulk = bulkCount * BS * BS;
    blocks = (totalElemsBulk + threads - 1) / threads;
    kernelBatchedMatrixAdd<<<blocks, threads>>>(rec.ee + rec.nmax * BS * BS,
                                                 rec.lsham + rec.nmax * BS * BS,
                                                 d_locham_bulk, bulkCount);
    cudaDeviceSynchronize();

    std::vector<const Complex*> psiB_bulk = buildConstPointerArray(rec.psi_b + rec.nmax * BS * BS, bulkCount);
    std::vector<const Complex*> locham_bulk = buildConstPointerArray(d_locham_bulk, bulkCount);
    std::vector<Complex*> hpsi_bulk = buildPointerArray(rec.hpsi + rec.nmax * BS * BS, bulkCount);
    batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                BS, BS, BS,
                &alpha,
                locham_bulk, BS,
                psiB_bulk, BS,
                &beta,
                hpsi_bulk, BS,
                bulkCount);
    cudaFree(d_locham_bulk);

    // --- END HOP PART ---

    // (2) Build the active atom list on the host.
    rec.irnum = 0;
    for (int i = 0; i < rec.kk; i++) {
        if (rec.izero[i] != 0) {
            rec.irlist[rec.irnum++] = i;
        }
    }
    // For the active atoms, we will need pointer arrays for psi_b, hpsi, and pmn_b.
    std::vector<const Complex*> psiB_active(rec.irnum);
    std::vector<Complex*> hpsi_active(rec.irnum);
    std::vector<Complex*> pmn_active(rec.irnum);
    for (int i = 0; i < rec.irnum; i++) {
        int idx = rec.irlist[i];
        psiB_active[i] = rec.psi_b + idx * BS * BS;
        hpsi_active[i] = rec.hpsi + idx * BS * BS;
        pmn_active[i] = rec.pmn_b + idx * BS * BS;
    }
    // For example, update pmn_b as: pmn_b = hpsi - pmn_b.
    // (Here we assume that pmn_b already contains a previous value; if not, you could
    // simply copy hpsi into pmn_b.)
    {
       Complex alpha_geam = MAKE_COMPLEX(1.0, 0.0);
       Complex beta_geam  = MAKE_COMPLEX(-1.0, 0.0);
       cublasStatus_t stat = cublasZgeamBatched(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            BS, BS,
                            &alpha_geam,
                            hpsi_active.data(), BS,
                            &beta_geam,
                            pmn_active.data(), BS,
                            pmn_active.data(), BS,
                            rec.irnum);
       if (stat != CUBLAS_STATUS_SUCCESS) {
           std::cerr << "cublasZgeamBatched failed" << std::endl;
           exit(EXIT_FAILURE);
       }
    }
    
    // (3) Accumulate a “summation” matrix over the active atoms:
    //     summ = sum_{i in active} (psi_b[i]^H * hpsi[i])
    // We compute for each active atom: temp_i = psi_b[i]^H * hpsi[i] (an 18x18 matrix)
    // and then reduce the batch on the host.
    Complex* d_temp;
    cudaMalloc(&d_temp, rec.irnum * BS * BS * sizeof(Complex));
    std::vector<const Complex*> psiB_active_conj = psiB_active; // will use CUBLAS_OP_C on psi_b
    std::vector<const Complex*> hpsi_active_const = hpsi_active;
    std::vector<Complex*> temp_ptrs = buildPointerArray(d_temp, rec.irnum);
    batchedGemm(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                BS, BS, BS,
                &alpha,
                psiB_active_conj, BS,
                hpsi_active_const, BS,
                &beta,
                temp_ptrs, BS,
                rec.irnum);
    // Now copy each 18×18 result back to host and sum.
    std::vector<Complex> summ(BS * BS, MAKE_COMPLEX(0.0, 0.0));
    std::vector<Complex> tempMatrix(BS * BS);
    for (int i = 0; i < rec.irnum; i++) {
         cudaMemcpy(tempMatrix.data(), d_temp + i * BS * BS,
                    BS * BS * sizeof(Complex), cudaMemcpyDeviceToHost);
         for (int j = 0; j < BS * BS; j++) {
             summ[j] = cuCadd(summ[j], tempMatrix[j]);
         }
    }
    cudaFree(d_temp);
    
    // (4) Save the summation matrix into atemp_b for this recursion step.
    // (For simplicity, we store it into atemp_b[0] – in a full code you would do this
    //  for each recursion step.)
    cudaMemcpy(rec.atemp_b, summ.data(), BS * BS * sizeof(Complex), cudaMemcpyHostToDevice);

    // --- CRECAL PART (Eigen–solving and updating psi_b, pmn_b) ---
    // For this example we diagonalize the summation matrix stored in atemp_b[0]
    // using cuSOLVER and then form b = U * sqrt(Λ) * Uᴴ and b_inv = U * sqrt(Λ)⁻¹ * Uᴴ.
    int matrixSize = BS;
    int lwork = 0, info = 0;
    // Query working space size (we use the upper triangle).
    cusolverDnZheevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_UPPER,
                                matrixSize, rec.atemp_b, matrixSize, nullptr, &lwork);
    Complex* d_work;
    cudaMalloc(&d_work, lwork * sizeof(Complex));
    int* devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    // Allocate eigenvalue array on host.
    std::vector<double> eigenvalues(matrixSize, 0.0);
    cusolverDnZheevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                      matrixSize, rec.atemp_b, matrixSize, eigenvalues.data(), d_work, lwork, nullptr, 0, devInfo);
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0) {
        std::cerr << "Eigen decomposition failed with info = " << info << std::endl;
        exit(EXIT_FAILURE);
    }
    // Now rec.atemp_b holds the eigenvectors (U).
    // On the host, build lam (diagonal with sqrt(eigenvalue)) and lam_inv.
    std::vector<Complex> U(matrixSize * matrixSize);
    cudaMemcpy(U.data(), rec.atemp_b, matrixSize * matrixSize * sizeof(Complex), cudaMemcpyDeviceToHost);
    std::vector<Complex> lam(matrixSize * matrixSize, MAKE_COMPLEX(0.0, 0.0));
    std::vector<Complex> lam_inv(matrixSize * matrixSize, MAKE_COMPLEX(0.0, 0.0));
    for (int i = 0; i < matrixSize; i++) {
         double sqrt_val = std::sqrt(eigenvalues[i]);
         lam[i * matrixSize + i] = MAKE_COMPLEX(sqrt_val, 0.0);
         lam_inv[i * matrixSize + i] = MAKE_COMPLEX(1.0 / sqrt_val, 0.0);
    }
    // Copy U, lam, and lam_inv to the device.
    Complex* d_U;  cudaMalloc(&d_U, matrixSize * matrixSize * sizeof(Complex));
    Complex* d_lam;  cudaMalloc(&d_lam, matrixSize * matrixSize * sizeof(Complex));
    Complex* d_lam_inv;  cudaMalloc(&d_lam_inv, matrixSize * matrixSize * sizeof(Complex));
    cudaMemcpy(d_U, U.data(), matrixSize * matrixSize * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lam, lam.data(), matrixSize * matrixSize * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lam_inv, lam_inv.data(), matrixSize * matrixSize * sizeof(Complex), cudaMemcpyHostToDevice);
    // Compute b = U * lam * Uᴴ and b_inv = U * lam_inv * Uᴴ.
    Complex* d_b;      cudaMalloc(&d_b, matrixSize * matrixSize * sizeof(Complex));
    Complex* d_b_inv;  cudaMalloc(&d_b_inv, matrixSize * matrixSize * sizeof(Complex));
    // For simplicity, we do these GEMMs with batch count = 1.
    std::vector<const Complex*> U_ptr = buildConstPointerArray(d_U, 1);
    std::vector<const Complex*> lam_ptr = buildConstPointerArray(d_lam, 1);
    std::vector<Complex*> tempMat_ptr = buildPointerArray(new Complex[matrixSize*matrixSize], 1); // temporary
    // Allocate temporary matrix on device.
    Complex* d_tempMat;  cudaMalloc(&d_tempMat, matrixSize * matrixSize * sizeof(Complex));
    {
       // temp = U * lam
       std::vector<const Complex*> U_arr = buildConstPointerArray(d_U, 1);
       std::vector<const Complex*> lam_arr = buildConstPointerArray(d_lam, 1);
       std::vector<Complex*> temp_arr = buildPointerArray(d_tempMat, 1);
       batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   U_arr, matrixSize,
                   lam_arr, matrixSize,
                   &beta,
                   temp_arr, matrixSize,
                   1);
    }
    {
       // b = temp * Uᴴ
       std::vector<const Complex*> temp_arr = buildConstPointerArray(d_tempMat, 1);
       std::vector<const Complex*> U_arr = buildConstPointerArray(d_U, 1);
       std::vector<Complex*> b_arr = buildPointerArray(d_b, 1);
       batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_C,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   temp_arr, matrixSize,
                   U_arr, matrixSize,
                   &beta,
                   b_arr, matrixSize,
                   1);
    }
    {
       // Compute b_inv = U * lam_inv * Uᴴ
       std::vector<const Complex*> lam_inv_arr = buildConstPointerArray(d_lam_inv, 1);
       std::vector<const Complex*> U_arr = buildConstPointerArray(d_U, 1);
       std::vector<Complex*> temp_arr = buildPointerArray(d_tempMat, 1);
       batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   U_arr, matrixSize,
                   lam_inv_arr, matrixSize,
                   &beta,
                   temp_arr, matrixSize,
                   1);
       std::vector<const Complex*> temp_arr_const = buildConstPointerArray(d_tempMat, 1);
       std::vector<const Complex*> U_arr_const = buildConstPointerArray(d_U, 1);
       std::vector<Complex*> b_inv_arr = buildPointerArray(d_b_inv, 1);
       batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_C,
                   matrixSize, matrixSize, matrixSize,
                   &alpha,
                   temp_arr_const, matrixSize,
                   U_arr_const, matrixSize,
                   &beta,
                   b_inv_arr, matrixSize,
                   1);
    }
    cudaFree(d_tempMat);
    cudaFree(d_U);  cudaFree(d_lam);  cudaFree(d_lam_inv);
    cudaFree(devInfo);  cudaFree(d_work);

    // (5) Finally, update psi_b and pmn_b for the active atoms using the new b and b_inv.
    // For example, one might do:
    //    psi_b[i] = pmn_b[i] * b_inv   and   pmn_b[i] = (saved psi_t[i]) * b
    // Here we show one batched GEMM for psi_b update.
    std::vector<const Complex*> pmn_active_const = buildConstPointerArray(rec.pmn_b, rec.irnum);
    std::vector<Complex*> psiB_update = buildPointerArray(rec.psi_b, rec.irnum); // using the same blocks as before
    // For the b_inv factor, note that it is the same for all atoms.
    // We “broadcast” its pointer by building a vector of identical pointers.
    std::vector<const Complex*> b_inv_bcast(rec.irnum, d_b_inv);
    batchedGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                BS, BS, BS,
                &alpha,
                pmn_active_const, BS,
                b_inv_bcast, BS,
                &beta,
                psiB_update, BS,
                rec.irnum);

    // Free temporary b matrices.
    cudaFree(d_b);  cudaFree(d_b_inv);

    // (At this point the recursion step is complete.)
}

// ----------------------------------------------------------------------
// Main: set up handles, allocate (dummy) arrays, and run the recursion.
// In a real code you would fill in the lattice/hamiltonian data properly.
// ----------------------------------------------------------------------
int main()
{
    // Initialize cuBLAS and cuSOLVER.
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // For this example, set dummy sizes.
    Recursion rec;
    rec.kk = 100;       // for example, 100 atoms
    rec.nmax = 20;      // 20 impurity atoms
    rec.lld = 10;       // 10 recursion steps

    // Allocate device arrays for psi_b, hpsi, pmn_b, atemp_b, b2temp_b.
    size_t atomMatrixSize = rec.kk * BS * BS * sizeof(Complex);
    cudaMalloc(&rec.psi_b, atomMatrixSize);
    cudaMalloc(&rec.hpsi,  atomMatrixSize);
    cudaMalloc(&rec.pmn_b, atomMatrixSize);
    cudaMalloc(&rec.hall,  rec.nmax * BS * BS * sizeof(Complex));
    cudaMalloc(&rec.ee,    (rec.kk - rec.nmax) * BS * BS * sizeof(Complex));
    cudaMalloc(&rec.lsham, rec.kk * BS * BS * sizeof(Complex));
    // For the recursion history arrays (assume one matrix per step).
    cudaMalloc(&rec.atemp_b, rec.lld * BS * BS * sizeof(Complex));
    cudaMalloc(&rec.b2temp_b, rec.lld * BS * BS * sizeof(Complex));

    // Allocate host arrays for izero and irlist.
    rec.izero = new int[rec.kk];
    rec.irlist = new int[rec.kk];
    // (For this example, mark all atoms as active.)
    for (int i = 0; i < rec.kk; i++) {
        rec.izero[i] = 1;
    }

    // (In a real code you would also initialize psi_b, hall, ee, lsham, etc.)

    // Run one recursion step (or loop over recursion steps as needed).
    runRecursion(rec, cublasHandle, cusolverHandle);

    // Clean up device memory.
    cudaFree(rec.psi_b);
    cudaFree(rec.hpsi);
    cudaFree(rec.pmn_b);
    cudaFree(rec.hall);
    cudaFree(rec.ee);
    cudaFree(rec.lsham);
    cudaFree(rec.atemp_b);
    cudaFree(rec.b2temp_b);

    // Clean up host memory.
    delete[] rec.izero;
    delete[] rec.irlist;

    // Destroy cuBLAS and cuSOLVER handles.
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    std::cout << "Recursion step completed successfully." << std::endl;
    return 0;
}
