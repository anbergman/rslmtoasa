// hop_b_cuda.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <complex>
#include <cassert>

using Complex = cuDoubleComplex;  // cuDoubleComplex is the double‐precision complex type
const int HBS = 18;               // Hamiltonian block size

//----------------------------------------------------------------------
// Helper device kernels
//----------------------------------------------------------------------

// Kernel to set an array of Complex numbers to zero.
__global__
void kernel_zero_complex(Complex* A, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

// Kernel to add two 18×18 matrices element‐wise: C = A + B
__global__
void kernel_add_matrices(const Complex* A, const Complex* B, Complex* C)
{
    int tid = threadIdx.x;
    if (tid < HBS*HBS) {
        C[tid] = cuCadd(A[tid], B[tid]);
    }
}

//----------------------------------------------------------------------
// Data structures (you must adapt these to your code)
//----------------------------------------------------------------------
struct Lattice {
    int kk;      // total number of atoms
    int nmax;    // number of impurity atoms (local region)
    // ... add additional arrays (e.g. nn, iz, etc.) and proper indexing
};

struct Hamiltonian {
    // Device pointers to the Hamiltonian blocks. We assume that the
    // arrays are stored in column‐major order and that each “block” is 18×18.
    Complex* hall;   // local Hamiltonian: size = (HBS*HBS*?) – see below
    Complex* ee;     // bulk Hamiltonian
    Complex* lsham;  // local spin–orbit corrections
};

struct Recursion {
    Lattice lattice;
    Hamiltonian hamiltonian;
    // Device pointers for the 3D arrays (each block is 18×18, with kk blocks):
    Complex* psi_b;    // [HBS x HBS x kk]
    Complex* hpsi;     // same dimensions
    Complex* pmn_b;    // same dimensions
    Complex* atemp_b;  // [HBS x HBS x (control.lld)]  — recursion “history”
    // Other arrays (e.g. izero, irlist) may be kept on the host.
    int* izero;      // host array of length kk
    int* irlist;     // host array (list of “active” atoms)
    int irnum;       // number of active atoms (set at the end)
};

//----------------------------------------------------------------------
// The CUDA version of hop_b.
// This function implements the equivalent of your Fortran hop_b routine.
// It uses cuBLAS for the GEMM calls and custom kernels for simple operations.
//----------------------------------------------------------------------
void hop_b_cuda(Recursion &rec, int ll, cublasHandle_t handle)
{
    // Assume rec.lattice.kk is the number of atoms.
    int numBlocks = rec.lattice.kk;
    int totalElems = numBlocks * HBS * HBS;
    int threadsPerBlock = 256;
    int numThreads = (totalElems + threadsPerBlock - 1) / threadsPerBlock;
    
    // 1. Zero out hpsi (device array)
    kernel_zero_complex<<<numThreads, threadsPerBlock>>>(rec.hpsi, totalElems);
    cudaDeviceSynchronize();
    
    // 2. Loop over impurity region atoms (i = 0 .. nmax-1)
    for (int i = 0; i < rec.lattice.nmax; i++) {
        if (rec.izero[i] != 0) {
            // In your Fortran code, you set:
            //   locham = hall(:,:,1,i) + lsham(:,:,ino)
            // Here we assume that the pointer for the i-th block of hall is
            // rec.hamiltonian.hall + i * HBS * HBS and that the proper index
            // “ino” is stored in your lattice (not shown).
            int ino = /* ... obtain atom type from your lattice data ... */ 0;
            
            // Allocate temporary device memory for locham (18×18)
            Complex* d_locham;
            cudaMalloc((void**)&d_locham, HBS * HBS * sizeof(Complex));
            
            // First, copy hall(:,:,1,i) into d_locham.
            // (In a production code, you might combine these two steps into one kernel.)
            cudaMemcpy(d_locham,
                       rec.hamiltonian.hall + i * HBS * HBS,
                       HBS * HBS * sizeof(Complex),
                       cudaMemcpyDeviceToDevice);
            
            // Now add lsham(:,:,ino) to d_locham.
            // Launch one block with HBS*HBS threads.
            kernel_add_matrices<<<1, HBS*HBS>>>(d_locham, rec.hamiltonian.lsham + ino * HBS * HBS, d_locham);
            cudaDeviceSynchronize();
            
            // Now perform: hpsi(:,:,i) += locham * psi_b(:,:,i)
            Complex alpha = make_cuDoubleComplex(1.0, 0.0);
            Complex beta  = make_cuDoubleComplex(1.0, 0.0);
            // Note: cuBLAS uses column-major ordering.
            cublasStatus_t stat = cublasZgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                HBS, HBS, HBS,
                                &alpha,
                                d_locham, HBS,
                                rec.psi_b + i * HBS * HBS, HBS,
                                &beta,
                                rec.hpsi + i * HBS * HBS, HBS);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasZgemm failed for impurity atom " << i << std::endl;
                exit(EXIT_FAILURE);
            }
            // (Optionally, add the neighbor contributions in a similar way.)
            
            cudaFree(d_locham);
        }
    }
    
    // 3. Process bulk atoms (i = nmax ... kk-1)
    for (int i = rec.lattice.nmax; i < rec.lattice.kk; i++) {
        // Similar to above, but using the bulk Hamiltonian rec.hamiltonian.ee and
        // the corresponding lsham block.
        int ih = /* ... obtain atom type from your lattice data ... */ 0;
        // Allocate temporary device memory for locham.
        Complex* d_locham;
        cudaMalloc((void**)&d_locham, HBS * HBS * sizeof(Complex));
        // Copy ee(:,:,1,ih) into d_locham.
        cudaMemcpy(d_locham,
                   rec.hamiltonian.ee + ih * HBS * HBS,
                   HBS * HBS * sizeof(Complex),
                   cudaMemcpyDeviceToDevice);
        // Add lsham(:,:,ih)
        kernel_add_matrices<<<1, HBS*HBS>>>(d_locham, rec.hamiltonian.lsham + ih * HBS * HBS, d_locham);
        cudaDeviceSynchronize();
        
        // GEMM: hpsi(:,:,i) += d_locham * psi_b(:,:,i)
        Complex alpha = make_cuDoubleComplex(1.0, 0.0);
        Complex beta  = make_cuDoubleComplex(1.0, 0.0);
        cublasStatus_t stat = cublasZgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                HBS, HBS, HBS,
                                &alpha,
                                d_locham, HBS,
                                rec.psi_b + i * HBS * HBS, HBS,
                                &beta,
                                rec.hpsi + i * HBS * HBS, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasZgemm failed for bulk atom " << i << std::endl;
            exit(EXIT_FAILURE);
        }
        // (Again, process neighbor contributions as in the Fortran code.)
        
        cudaFree(d_locham);
    }
    
    // 4. Build the “irlist” array and update rec.irnum.
    // This is a reduction over the host array rec.izero[].
    rec.irnum = 0;
    for (int i = 0; i < rec.lattice.kk; i++) {
        if (rec.izero[i] != 0) {
            rec.irlist[rec.irnum++] = i;
        }
    }
    
    // 5. Finally, loop over the active atoms and update pmn_b.
    // (For each active atom, perform a GEMM on 18×18 matrices.)
    for (int k = 0; k < rec.irnum; k++) {
        int i = rec.irlist[k];
        // Here we perform:
        //    pmn_b(:,:,i) = hpsi(:,:,i) - pmn_b(:,:,i)
        // followed by: summ += psi_b(:,:,i)ᶜ * hpsi(:,:,i)
        // In our GPU code you might combine these using a custom kernel
        // or call cuBLAS for the GEMM (using CUBLAS_OP_C for conjugate transpose).
        // For example:
        Complex alpha = make_cuDoubleComplex(1.0, 0.0);
        Complex beta  = make_cuDoubleComplex(-1.0, 0.0);
        cublasStatus_t stat = cublasZgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                HBS, HBS,
                                &alpha,
                                rec.hpsi + i * HBS * HBS, HBS,
                                &beta,
                                rec.pmn_b + i * HBS * HBS, HBS,
                                rec.pmn_b + i * HBS * HBS, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasZgeam failed for active atom " << i << std::endl;
            exit(EXIT_FAILURE);
        }
        // And add to the “summation” matrix (here we assume summ is allocated on device).
        // For the GEMM (or reduction) you could accumulate into a device variable.
        // (This step is left as an exercise—you may wish to use a batched approach.)
    }
    
    // 6. Copy the final summation to atemp_b(:,:,ll)
    // (Assume summ is stored in a temporary device array "d_summ" of size HBS*HBS.)
    // cudaMemcpy(rec.atemp_b + ll*HBS*HBS, d_summ, HBS*HBS*sizeof(Complex), cudaMemcpyDeviceToDevice);
}
