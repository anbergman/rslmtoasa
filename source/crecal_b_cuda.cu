// crecal_b_cuda.cu
#include <cusolverDn.h>
// (Assume the same headers and data structures as above.)

void crecal_b_cuda(Recursion &rec, cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle)
{
    // Variables for the eigen–solver and temporary matrices.
    Complex dum[HBS*HBS];
    Complex b[HBS*HBS];
    Complex b_inv[HBS*HBS];
    Complex u[HBS*HBS];
    Complex lam[HBS*HBS] = { 0 };   // Will be diagonal
    Complex lam_inv[HBS*HBS] = { 0 };
    double ev[HBS];               // Eigenvalues (real)
    
    int nm1 = /* control.lld */  /* number of recursion steps */ 0;
    int llmax = /* control.lld */ 0;
    
    // For each recursion step (ll = 0, 1, ..., nm1-1)
    for (int ll = 0; ll < nm1; ll++) {
        // Reset atemp_b(:,:,ll) to zero.
        int total = HBS * HBS;
        int threadsPerBlock = 256;
        int numThreads = (total + threadsPerBlock - 1) / threadsPerBlock;
        kernel_zero_complex<<<numThreads, threadsPerBlock>>>(rec.atemp_b + ll*HBS*HBS, total);
        cudaDeviceSynchronize();
        
        // 1. Call hop_b_cuda to compute A_n = H|psi_n>
        hop_b_cuda(rec, ll, cublasHandle);
        
        // 2. Save psi_b into a temporary array (psi_t).
        // You may want to allocate a temporary device array for psi_t.
        // For simplicity we assume that psi_t is allocated with the same dimensions as psi_b.
        // (A batched device-to-device copy of many 18×18 matrices.)
        
        // 3. Copy the current b2temp_b (which holds the previous B_n) into a host/device temporary.
        // (Assume that rec.b2temp_b is stored similarly to psi_b.)
        
        // 4. Now update pmn_b: for each active atom, do:
        //    pmn_b(:,:,i) = hpsi(:,:,i) - psi_b(:,:,i)*atemp_b(:,:,ll)
        // and then accumulate: sum_b += (pmn_b(:,:,i))ᶜ * (pmn_b(:,:,i))
        // Use cuBLAS (or batched GEMM) for these operations.
        // (The following is a “per–atom” loop for clarity.)
        Complex* d_sum;  // allocate a device array for the summation (18×18)
        cudaMalloc((void**)&d_sum, HBS * HBS * sizeof(Complex));
        kernel_zero_complex<<<numThreads, threadsPerBlock>>>(d_sum, HBS*HBS);
        cudaDeviceSynchronize();
        
        for (int idx = 0; idx < rec.irnum; idx++) {
            int i = rec.irlist[idx];
            // pmn_b(:,:,i) = hpsi(:,:,i) - psi_b(:,:,i)*atemp_b(:,:,ll)
            // This can be done by a GEMM followed by an element-wise subtraction.
            Complex alpha = make_cuDoubleComplex(-1.0, 0.0);
            Complex beta  = make_cuDoubleComplex(1.0, 0.0);
            // First update pmn_b: pmn_b = hpsi - psi_b * atemp_b.
            cublasStatus_t stat = cublasZgemm(cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HBS, HBS, HBS,
                            &alpha,
                            rec.psi_b + i * HBS * HBS, HBS,
                            rec.atemp_b + ll * HBS * HBS, HBS,
                            &beta,
                            rec.pmn_b + i * HBS * HBS, HBS);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasZgemm failed in crecal_b for atom " << i << std::endl;
                exit(EXIT_FAILURE);
            }
            
            // Now accumulate: d_sum += (pmn_b(:,:,i))ᶜ * (pmn_b(:,:,i))
            // Use cublasZgemm with CUBLAS_OP_C on the left.
            alpha = make_cuDoubleComplex(1.0, 0.0);
            beta  = make_cuDoubleComplex(1.0, 0.0);
            stat = cublasZgemm(cublasHandle,
                               CUBLAS_OP_C, CUBLAS_OP_N,
                               HBS, HBS, HBS,
                               &alpha,
                               rec.pmn_b + i * HBS * HBS, HBS,
                               rec.pmn_b + i * HBS * HBS, HBS,
                               &beta,
                               d_sum, HBS);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Accumulation GEMM failed in crecal_b for atom " << i << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        // 5. Use cuSOLVER to diagonalize the 18×18 hermitian matrix d_sum.
        //    (The eigen–solver returns eigenvalues ev[0..HBS-1] and eigen–vectors in u.)
        // For example:
        int lwork = 0;
        int info = 0;
        // First, query workspace size.
        cusolverDnZheevd_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    HBS, d_sum, HBS, ev, &lwork);
        Complex* d_work;
        cudaMalloc((void**)&d_work, lwork * sizeof(Complex));
        int* devInfo;
        cudaMalloc((void**)&devInfo, sizeof(int));
        
        cusolverDnZheevd(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                          HBS, d_sum, HBS, ev, d_work, lwork, nullptr, 0, devInfo);
        cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (info != 0) {
            std::cerr << "cuSOLVER eigenvalue decomposition failed with info = " << info << std::endl;
            exit(EXIT_FAILURE);
        }
        // Now, d_sum holds the eigen–vectors (we call that u). Copy them into u.
        cudaMemcpy(u, d_sum, HBS * HBS * sizeof(Complex), cudaMemcpyDeviceToHost);
        
        // 6. Form lam (diagonal with sqrt(ev)) and its inverse.
        for (int i = 0; i < HBS; i++) {
            double lambda = sqrt(ev[i]);
            lam[i*HBS + i] = make_cuDoubleComplex(lambda, 0.0);
            lam_inv[i*HBS + i] = make_cuDoubleComplex(1.0/lambda, 0.0);
        }
        
        // 7. Compute b = u * lam * uᴴ and b_inv = u * lam_inv * uᴴ.
        // (This can be done as two GEMMs each.)
        Complex alpha = make_cuDoubleComplex(1.0, 0.0);
        Complex beta  = make_cuDoubleComplex(0.0, 0.0);
        // For b:
        // First compute an intermediate: dum = u * lam.
        stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HBS, HBS, HBS,
                           &alpha,
                           u, HBS,
                           lam, HBS,
                           &beta,
                           dum, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
        // Then b = dum * uᴴ.
        stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
                           HBS, HBS, HBS,
                           &alpha,
                           dum, HBS,
                           u, HBS,
                           &beta,
                           b, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
        
        // Similarly, compute b_inv = u * lam_inv * uᴴ.
        stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HBS, HBS, HBS,
                           &alpha,
                           u, HBS,
                           lam_inv, HBS,
                           &beta,
                           dum, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
        stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C,
                           HBS, HBS, HBS,
                           &alpha,
                           dum, HBS,
                           u, HBS,
                           &beta,
                           b_inv, HBS);
        if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
        
        // 8. Update psi_b and pmn_b by applying b and b_inv.
        for (int idx = 0; idx < rec.irnum; idx++) {
            int i = rec.irlist[idx];
            // psi_b(:,:,i) = pmn_b(:,:,i) * b_inv   (for example)
            stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HBS, HBS, HBS,
                               &alpha,
                               rec.pmn_b + i * HBS * HBS, HBS,
                               b_inv, HBS,
                               &beta,
                               rec.psi_b + i * HBS * HBS, HBS);
            if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
            
            // And: pmn_b(:,:,i) = psi_t(:,:,i) * b  (for example)
            stat = cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                               HBS, HBS, HBS,
                               &alpha,
                               /* psi_t for atom i (assumed stored) */, HBS,
                               b, HBS,
                               &beta,
                               rec.pmn_b + i * HBS * HBS, HBS);
            if (stat != CUBLAS_STATUS_SUCCESS) { /* error */ }
        }
        
        // Free temporary workspace for this recursion step.
        cudaFree(d_sum);
        cudaFree(d_work);
        cudaFree(devInfo);
        
        // (Optionally, update rec.b2temp_b for this ll step.)
    }
    
    // Finally, store the last sum into rec.b2temp_b(:,:,llmax)
}
