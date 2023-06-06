#ifndef __SVD_HPP__
#define __SVD_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>
#include <vector>
#include <iostream>

#include "dpc_common.hpp"
#include "svd_helper.hpp"
#include "memory_direct_access.hpp"
#include "matmul_naive.hpp"
#include "StreamingFullQRD.hpp"
#include "pipe_mux.hpp"
#include "diagonal_convergence.hpp"
#include "accumulator.hpp"
#include "orthogonalizer.hpp"
#include "post_process.hpp"

// pipes 
class APipe;
class AtPipe;
class AtAPipe;
class MatmulItersPipe;
class RQPipeW;
class RQPipeR;
class QRD_APipe;
class QRD_Q_Pipe_2dup;
class QRD_R_Pipe_2checker;
class QRD_Q_Pipe_2accum;
class QRD_Q_Pipe_2matmul;
class QRD_R_Pipe_2matmul;
class QRD_R_Pipe_2S;
class QRD_iterations_Pipe;
class SVD_V_Pipe;
// class SVD_V_ON_Pipe;
class SVD_V_Pipe2dma;
// class SVD_V_Pipe2post;
// class SVD_S_Pipe2dup;
// class SVD_S_Pipe2V;
class SVD_S_Pipe2dma;
// class SVD_U_Pipe;
class SVD_U_ON_Pipe;
class SVD_Converge_Pipe;
class Post_A_Pipe;
// class Post_AV_Pipe;
// kernels
class MATMULDDR2PipeA;
class MATMULDDR2PipeAt;
class MATMUL;
class MUX_2TO1;
class ITER_QRD;
class MATMUL_RQ;
class DEMUX_1TO2_R;
class QRDSaveAAt;
class QRDReadAAt;
class QRDSaveRQ;
class QRDReadRQ;
class SVDUPipe2DDR;
class SVDSPipe2DDR;
class SVDVPipe2DDR;
class ConvChecker;
class CollectIterationCount;
class QAccumulator;
class QDuplicator;
// class VOrthogonalizer;
// class UOrthogonalizer;
// class SVDSBuilder;
// class SDuplicator;
// class VDuplicator;
// class MatMul_AV;
// class SVDUBuilder;
class SVDPostDDR2PipeA;
class SVDPostProcess;


template <unsigned rows_a,  // row size of input A
          unsigned cols_a,  // col size of input A
          bool is_complex,
          typename T,       // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
         >
void singularValueDecomposition(
        std::vector<TT> &a_matrix,  // input matrix A (col maj)
        std::vector<TT> &u_matrix,  // left singular vectors, U (col maj)
        std::vector<TT> &s_values,  // list of singular values, S
        std::vector<TT> &v_matrix,  // right singular vectors, V (row maj)
        sycl::queue &q
)
{
    // set up useful constants 
    constexpr int aMatrixSize = rows_a * cols_a;
    constexpr int qMatrixSize = cols_a * cols_a;
    constexpr int rMatrixSize = rows_a * cols_a;
    constexpr int uMatrixSize = rows_a * rows_a;
    constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

    // make At (A transpose)
    std::vector<TT> at_matrix(aMatrixSize);
    for (int row = 0; row < rows_a; row ++)
    {
        for (int col = 0; col < cols_a; col ++)
        {
            at_matrix[row * cols_a + col] = a_matrix[col * rows_a + row];
        }
    }

    // pipes for inter-kernels communication
    using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;
    using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
    using AtMatrixPipe = sycl::ext::intel::pipe<AtPipe, PipeType, 3>;
    using AtAMatrixPipe = sycl::ext::intel::pipe<AtAPipe, PipeType, 3>;
    using AtAMatmulItersPipe = sycl::ext::intel::pipe<MatmulItersPipe, int, 3>;
    using RQMatrixPipeW = sycl::ext::intel::pipe<RQPipeW, PipeType, 3>;
    // using RQMatrixPipeR = sycl::ext::intel::pipe<RQPipeR, PipeType, 3>;
    using QRD_AMatrixPipe = sycl::ext::intel::pipe<QRD_APipe, PipeType, 3>;
    using QMatrixPipe2Dup = sycl::ext::intel::pipe<QRD_Q_Pipe_2dup, PipeType, 3>;
    using QMatrixPipe2Matmul = sycl::ext::intel::pipe<QRD_Q_Pipe_2matmul, PipeType, 3>;
    using QMatrixPipe2Accumulator = sycl::ext::intel::pipe<QRD_Q_Pipe_2accum, PipeType, 3>;
    using RMatrixPipe2Check = sycl::ext::intel::pipe<QRD_R_Pipe_2checker, PipeType, 3>;
    using RMatrixPipe2Matmul = sycl::ext::intel::pipe<QRD_R_Pipe_2matmul, PipeType, 3>;
    using RMatrixPipe2S = sycl::ext::intel::pipe<QRD_R_Pipe_2S, PipeType, 3>;
    using SMatrixPipe2Dma = sycl::ext::intel::pipe<SVD_S_Pipe2dma, PipeType, 3>;
    using IterationsPipe = sycl::ext::intel::pipe<QRD_iterations_Pipe, int, 3>;
    using VMatrixPipe = sycl::ext::intel::pipe<SVD_V_Pipe, PipeType, 3>;
    // using VOrthNormPipe = sycl::ext::intel::pipe<SVD_V_ON_Pipe, PipeType, 3>;
    using VPipe2dma = sycl::ext::intel::pipe<SVD_V_Pipe2dma, PipeType, 3>;
    // using VPipe2post = sycl::ext::intel::pipe<SVD_V_Pipe2post, PipeType, 3>;
    using UOrthNormPipe = sycl::ext::intel::pipe<SVD_U_ON_Pipe, PipeType, 3>;
    // using SPipe2dup = sycl::ext::intel::pipe<SVD_S_Pipe2dup, PipeType, 3>;
    // using SPipe2V = sycl::ext::intel::pipe<SVD_S_Pipe2V, PipeType, 3>;
    // using UPipe = sycl::ext::intel::pipe<SVD_U_Pipe, PipeType, 3>;
    using ConvergePipe = sycl::ext::intel::pipe<SVD_Converge_Pipe, bool, 2>;
    using PostAPipe = sycl::ext::intel::pipe<Post_A_Pipe, PipeType, 3>;
    // using PostAVPipe = sycl::ext::intel::pipe<Post_AV_Pipe, PipeType, 3>;

    // Allocate FPGA  memory.
    TT *a_device = sycl::malloc_device<TT>(aMatrixSize, q);
    if (a_device == nullptr) {
        std::cout << "Malloc A matrix USM failed" << std::endl;
        return;
    }
    TT *at_device = sycl::malloc_device<TT>(aMatrixSize, q);
    if (at_device == nullptr) {
        std::cout << "Malloc At matrix USM failed" << std::endl;
        return;
    }
    TT *v_device = sycl::malloc_device<TT>(qMatrixSize, q);
    TT *r_device = sycl::malloc_device<TT>(rMatrixSize, q);
    TT *u_device = sycl::malloc_device<TT>(uMatrixSize, q);
    int *iters_device = sycl::malloc_device<int>(1, q);

    // copy matrix datas to FPGA DDR
    q.memcpy(a_device, a_matrix.data(), aMatrixSize * sizeof(TT))
        .wait();
    q.memcpy(at_device, at_matrix.data(), aMatrixSize * sizeof(TT))
      .wait();
    std::cout << "Start kernels" << std::endl;
    // DMA Producer kernel for matrix A
    auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<MATMULDDR2PipeA>([=]() [[intel::kernel_args_restrict]] {
        MatrixReadFromDDRToPipeColMaj<TT, rows_a, cols_a, kNumElementsPerDDRBurst,
                                AMatrixPipe>(a_device);
        });
    });

    // DMA Producer kernel for matrix At
    q.submit([&](sycl::handler &h) {
    h.single_task<MATMULDDR2PipeAt>([=]() [[intel::kernel_args_restrict]] {
        MatrixReadFromDDRToPipeColMaj<TT, cols_a, rows_a, kNumElementsPerDDRBurst,
                                AtMatrixPipe>(at_device);
        });
    });

    // kernel calculate A @ At
    sycl::event matmul_event = q.single_task<MATMUL>(
        fpga_linalg::NaiveMatmul<TT, is_complex, cols_a, rows_a, rows_a, cols_a,
                                kNumElementsPerDDRBurst, AtMatrixPipe,
                                AMatrixPipe, AtAMatrixPipe, AtAMatmulItersPipe>{1}
    );

    // muxing between RQ and AtA
    q.single_task<MUX_2TO1>(
        PipeMux2to1<PipeType, AtAMatmulItersPipe, RQMatrixPipeW, AtAMatrixPipe, QRD_AMatrixPipe>{}
    );

    // QR iterations -----------------------------------------------------------------
    int QR_MAX_ITERATIONS = 99;

    // kernel QR decompose AAt
    sycl::event rq_qrd_event = q.single_task<ITER_QRD>(
            fpga_linalg::StreamingFullQRD<TT, is_complex, cols_a, cols_a, 110, 
                                kNumElementsPerDDRBurst, QRD_AMatrixPipe, QMatrixPipe2Dup, 
                                RMatrixPipe2Check>{QR_MAX_ITERATIONS+1}
    );

    // convergency checker
    q.single_task<ConvChecker>(
        fpga_svd::DiagonalConvergence<TT, is_complex, cols_a, cols_a, kNumElementsPerDDRBurst, 
                                RMatrixPipe2Check, RMatrixPipe2Matmul, RMatrixPipe2S, ConvergePipe>
                                {QR_MAX_ITERATIONS+1, EPSILON, MAX_CONVERGENCY_ERROR}       // 2% convergency error, 1e-6 can considered zero
    );

    // Q duplicator
    q.single_task<QDuplicator>(
        PipeDuplicator2x<TT, QMatrixPipe2Dup, QMatrixPipe2Matmul, QMatrixPipe2Accumulator>{}
    );

    // QR iteration: R @ Q
    sycl::event rq_matmul_event = q.single_task<MATMUL_RQ>(
        fpga_linalg::NaiveMatmul<TT, is_complex, cols_a, cols_a, cols_a, cols_a,
                        kNumElementsPerDDRBurst, RMatrixPipe2Matmul,
                        QMatrixPipe2Matmul, RQMatrixPipeW, IterationsPipe>{QR_MAX_ITERATIONS}
    );

    // Q accumulator
    sycl::event q_accumulate_event = q.single_task<QAccumulator>(
        fpga_svd::Accumulator_Mult<TT, is_complex, cols_a, cols_a,
                        kNumElementsPerDDRBurst, QMatrixPipe2Accumulator,
                        VMatrixPipe, ConvergePipe>{QR_MAX_ITERATIONS}
    );

    // // V orthogonazer
    // sycl::event v_orthogonalize_event = q.single_task<VOrthogonalizer>(
    //     Orthogonazer<TT, is_complex, cols_a, cols_a, 110, 
    //                     kNumElementsPerDDRBurst, VMatrixPipe, 
    //                     VOrthNormPipe>{1}
    // );

    // // V duplicator
    // q.single_task<VDuplicator>(
    //     PipeDuplicator2x<TT, VOrthNormPipe, VPipe2post, VPipe2dma>{}
    // );

    // // Build S from R
    // sycl::event s_bulid_event = q.single_task<SVDSBuilder>(
    //     fpga_svd::SBuilder<TT, is_complex, cols_a, cols_a, rows_a, cols_a, 
    //                     kNumElementsPerDDRBurst, RMatrixPipe2S, SPipe2dup>{1}
    // );

    // // S duplicator
    // q.single_task<SDuplicator>(
    //     PipeDuplicator2x<TT, SPipe2dup, SPipe2V, SMatrixPipe2Dma>{}
    // );

    // read in A (again)
    q.submit([&](sycl::handler &h) {
    h.single_task<SVDPostDDR2PipeA>([=]() [[intel::kernel_args_restrict]] {
        MatrixReadFromDDRToPipeColMaj<TT, rows_a, cols_a, kNumElementsPerDDRBurst,
                                PostAPipe>(a_device);
        });
    });

    // // kernel calculate A @ V
    // q.single_task<MatMul_AV>(
    //     fpga_linalg::NaiveMatmulOnce<TT, is_complex, rows_a, cols_a, cols_a, cols_a,
    //                             kNumElementsPerDDRBurst, PostAPipe,
    //                             VPipe2post, PostAVPipe>{}
    // );

    // // Build U from AV and S
    // sycl::event u_bulid_event = q.single_task<SVDUBuilder>(
    //     fpga_svd::UBuilder<TT, is_complex, rows_a, cols_a, rows_a, cols_a, 
    //                     kNumElementsPerDDRBurst, PostAVPipe, SPipe2V, UPipe>{}
    // );

    // // U orthogonazer
    // sycl::event u_orthogonalize_event = q.single_task<UOrthogonalizer>(
    //     Orthogonazer<TT, is_complex, rows_a, rows_a, 110, 
    //                     kNumElementsPerDDRBurst, UPipe, 
    //                     UOrthNormPipe>{1}
    // );

    // SVD Post process
    q.single_task<SVDPostProcess>(
        fpga_svd::PostProcess<TT, is_complex, rows_a, cols_a,
                                kNumElementsPerDDRBurst, 
                                PostAPipe, RMatrixPipe2S, VMatrixPipe, 
                                UOrthNormPipe, SMatrixPipe2Dma, VPipe2dma>{}
                                
    );

    // DMA Consumer kernel for the result S matrix
    sycl::event s_out_event =
        q.single_task<SVDSPipe2DDR>([=]() [[intel::kernel_args_restrict]] {
            MatrixReadPipeToDDRColMaj<TT, rows_a, cols_a, kNumElementsPerDDRBurst,
                                SMatrixPipe2Dma>(r_device);
    });

    // DMA Consumer kernel V matrix
    sycl::event v_out_event =
         q.single_task<SVDVPipe2DDR>([=]() [[intel::kernel_args_restrict]] {
            MatrixReadPipeToDDRColMaj<TT, cols_a, cols_a, kNumElementsPerDDRBurst,
                                VPipe2dma>(v_device);
    });

    // DMA Consumer kernel U matrix
    sycl::event u_out_event =
         q.single_task<SVDUPipe2DDR>([=]() [[intel::kernel_args_restrict]] {
            MatrixReadPipeToDDRColMaj<TT, rows_a, rows_a, kNumElementsPerDDRBurst,
                                UOrthNormPipe>(u_device);
    });

    // Collecting the iteration count
    q.single_task<CollectIterationCount>( 
        CollectPipeToDDR<int, IterationsPipe>{iters_device}
    );
    // wait for consumer kernels (the last kernel to finish)
    s_out_event.wait();
    v_out_event.wait();
    u_out_event.wait();
    std::cout << "End kernels" << std::endl;
    q.memcpy(s_values.data(), r_device, rMatrixSize * sizeof(TT))
        .wait();
    q.memcpy(v_matrix.data(), v_device, qMatrixSize * sizeof(TT))
        .wait();
    q.memcpy(u_matrix.data(), u_device, uMatrixSize * sizeof(TT))
        .wait();
    
    int total_iteration[1];
    q.memcpy(total_iteration, iters_device, 1 * sizeof(int))
        .wait();
    std::cout << "SVD Done, total iterations: " << total_iteration[0] << std::endl;
    if (total_iteration[0]>= QR_MAX_ITERATIONS) std::cout << "Max iteration limit reached" << std::endl;
    
    sycl::free(a_device, q);
    sycl::free(at_device, q);
    sycl::free(r_device, q);
    // sycl::free(iters_device, q);
    sycl::free(v_device, q);
    sycl::free(u_device, q);
    std::cout << "Done freeing" <<std::endl;
}



#endif /* __SVD_HPP__ */
