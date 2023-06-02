#ifndef __MATMUL_NAIVE__HPP__
#define __MATMUL_NAIVE__HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  Matrix multiplication
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int A_rows,       // Number of rows in the input matrices
          int A_columns,    // Number of columns in the input matrices
          int B_rows,       // Number of rows in the input matrices
          int B_columns,    // Number of columns in the input matrices
          int pipe_size,    // Number of elements read/write per pipe operation
                            // to read the input matrix
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename BIn,     // B matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename MMOut,    // MM matrix output pipe, send one elements to the
                            // pipe with each write.
          typename IterationsFinished
          >
struct NaiveMatmul {
    int iteration_count;

    void operator()() const {
        // Functional assertions
        static_assert((A_rows >= 4) && (A_columns >= 4) && (B_rows >= 4) && (B_columns >= 4),
                    "Only matrices of size 4x4 and over are supported");
        // static_assert(rows == columns,
        //               "Only written for square matrices, can be extended");
        static_assert(A_columns == B_rows,
                    "Illegal sizes for matrix multiplication");
        static_assert(pipe_size >= 1,
                    "The pipe must be able to contain at least one element");
        // PRINTF("Start Matmul\n")
        // Set the computation type to T or ac_complex<T> depending on the value
        // of is_complex
        using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

        // Compute Cholesky decompositions as long as matrices are given as inputs
        // Break memories up to store pipe_size elements per bank
        constexpr short kBankwidth = pipe_size * sizeof(TT);
        constexpr unsigned short aNumBanks = A_rows / pipe_size;
        constexpr unsigned short bNumBanks = B_rows / pipe_size;
        constexpr unsigned short mmNumBanks = A_rows / pipe_size;

        // When specifying numbanks for a memory, it must be a power of 2.
        // Unused banks will be automatically optimized away.
        constexpr short aNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(aNumBanks));
        constexpr short bNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(bNumBanks));
        constexpr short mmNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(mmNumBanks));
            
        for (int iteration = 0; iteration < iteration_count; iteration ++) {

                [[intel::numbanks(aNumBanksNextPow2)]]  // NO-FORMAT: Attribute
                [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
                [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
                [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
                TT a_load[A_rows][A_columns];

                [[intel::numbanks(bNumBanksNextPow2)]]  // NO-FORMAT: Attribute
                [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
                [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
                [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
                TT b_load[B_rows][B_columns];

                [[intel::numbanks(mmNumBanksNextPow2)]]  // NO-FORMAT: Attribute
                [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
                [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
                [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
                TT mm_result[B_columns][A_rows];

                // Copy a matrix from the pipe to a local memory
                // Number of pipe reads of pipe_size required to read a full column
                constexpr int aExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
                constexpr int bExtraIteration = ((B_rows % pipe_size) != 0) ? 1 : 0;
                constexpr int mmExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
                constexpr int aLoopIterPerColumn = (A_rows / pipe_size) + aExtraIteration;
                constexpr int bLoopIterPerColumn = (B_rows / pipe_size) + bExtraIteration;
                constexpr int mmLoopIterPerColumn = (A_rows / pipe_size) + mmExtraIteration;
                // Number of pipe reads of pipe_size to read all the matrices
                constexpr int aLoopIter = aLoopIterPerColumn * A_columns;
                constexpr int bLoopIter = bLoopIterPerColumn * B_columns;
                constexpr int mmLoopIter = mmLoopIterPerColumn * B_columns;
                // Size in bits of the loop iterator over kLoopIter iterations
                constexpr int aLoopIterBitSize =
                    fpga_tools::BitsForMaxValue<aLoopIter + 1>();
                constexpr int bLoopIterBitSize =
                    fpga_tools::BitsForMaxValue<bLoopIter + 1>();
                constexpr int mmLoopIterBitSize =
                    fpga_tools::BitsForMaxValue<mmLoopIter + 1>();

                // load matrix A from pipe
#ifndef LARGE_MATRIX
                [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
                for (ac_int<aLoopIterBitSize, false> li = 0; li < aLoopIter; li++) {
                    fpga_tools::NTuple<TT, pipe_size> pipe_read_a = AIn::read();

                    int write_idx_a = li % aLoopIterPerColumn;
                    fpga_tools::UnrolledLoop<aLoopIterPerColumn>([&](auto k) {
                        fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                            
                            if constexpr (k * pipe_size + t < A_rows) {
                                if (write_idx_a == k) {
                                    a_load[k * pipe_size + t][li / aLoopIterPerColumn] =
                                        pipe_read_a.template get<t>();
                                }
                            }

                            // Delay data signals to create a vine-based data distribution
                            // to lower signal fanout.
                            pipe_read_a.template get<t>() =
                                sycl::ext::intel::fpga_reg(pipe_read_a.template get<t>());
                        });
                    
                        write_idx_a = sycl::ext::intel::fpga_reg(write_idx_a);
                    });
                }

                // load matrix B from pipe
#ifndef LARGE_MATRIX
                [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
                for (ac_int<bLoopIterBitSize, false> li = 0; li < bLoopIter; li++) {
                    fpga_tools::NTuple<TT, pipe_size> pipe_read_b = BIn::read();

                    int write_idx_b = li % bLoopIterPerColumn;

                    fpga_tools::UnrolledLoop<bLoopIterPerColumn>([&](auto k) {
                        fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {

                            if constexpr (k * pipe_size + t < B_rows) {
                                if (write_idx_b == k) {
                                    b_load[k * pipe_size + t][li / bLoopIterPerColumn] =
                                        pipe_read_b.template get<t>();
                                }
                            }

                            // Delay data signals to create a vine-based data distribution
                            // to lower signal fanout.
                            pipe_read_b.template get<t>() =
                                sycl::ext::intel::fpga_reg(pipe_read_b.template get<t>());
                        });
                        
                        write_idx_b = sycl::ext::intel::fpga_reg(write_idx_b);
                    });
                }

                // Compute the matrix product
                for (int row = 0; row < A_rows; row++) {
                    for (int column = 0; column < B_columns; column++) {
                        TT dot_prod{0};
                        fpga_tools::UnrolledLoop<A_columns>([&](auto k) {
                            // Assume dot_prods the B matrix was given transposed, otherwise it need to
                            // be transposed.
                            dot_prod = sycl::ext::intel::fpga_reg(dot_prod) +
                                    a_load[row][k] * b_load[k][column];
                        });
                        mm_result[row][column] = dot_prod;
                    }
                }

                // // debug ========================
                // for (int row = 0; row < A_rows; row ++)
                // {
                //     for (int col = 0; col < B_columns; col ++)
                //     {
                //         // PRINTF("%.2f ", mm_result[row][col]);
                //         PRINTF("%.2f ", a_load[row][col]);
                //     }
                //     PRINTF("\n");
                // }
                // PRINTF("---------------------\n");
                // // ==============================

                // Copy the result matrix to the output pipe (in col maj)
                [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
                for (ac_int<mmLoopIterBitSize, false> li = 0; li < mmLoopIter; li++) {
                    int column_iter = li % mmLoopIterPerColumn;
                    bool get[mmLoopIterPerColumn];
                    fpga_tools::UnrolledLoop<mmLoopIterPerColumn>([&](auto k) {
                    get[k] = column_iter == k;
                    column_iter = sycl::ext::intel::fpga_reg(column_iter);
                    });
            
                    fpga_tools::NTuple<TT, pipe_size> pipe_write;
                    fpga_tools::UnrolledLoop<mmLoopIterPerColumn>([&](auto t) {
                        fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                            if constexpr (t * pipe_size + k < A_rows) {
                                pipe_write.template get<k>() =
                                    get[t] ? mm_result[t * pipe_size + k][li / mmLoopIterPerColumn]
                                            : sycl::ext::intel::fpga_reg(
                                                pipe_write.template get<k>());
                                // PRINTF("%.2f ", pipe_write.template get<k>());
                            }
                        });
                    });
                    MMOut::write(pipe_write);
                }
                IterationsFinished::write(iteration+1);
                // PRINTF("Matmul iteration %d done\n", iteration);
        } // end of for loop (iterations)
    }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __MATMUL_NAIVE_HPP__ */