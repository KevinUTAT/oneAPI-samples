#ifndef __ACCUMULATOR_B_HPP__ 
#define __ACCUMULATOR_B_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_svd {
/*
*  Matrix multiply-accumilator
*  A = A @ B 
*/

template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int B_rows,       // Number of rows in the input matrices
          int B_columns,    // Number of columns in the input matrices
          int pipe_size,    // Number of elements read/write per pipe operation
                            // to read the input matrix
          typename BIn,     // B matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename MMOut,    // MM matrix output pipe, send one elements to the
                            // pipe with each write.
          typename OutEnable_pipe
          >
struct Accumulator_Mult {
    int iteration_count;

    void operator()() const {
        // Functional assertions
        static_assert((B_rows >= 4) && (B_columns >= 4),
                    "Only matrices of size 4x4 and over are supported");
        // static_assert(rows == columns,
        //               "Only written for square matrices, can be extended");
        static_assert(B_columns == B_rows,
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
        constexpr unsigned short bNumBanks = B_rows / pipe_size;
        constexpr unsigned short mmNumBanks = B_rows / pipe_size;

        // When specifying numbanks for a memory, it must be a power of 2.
        // Unused banks will be automatically optimized away.
        constexpr short bNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(bNumBanks));
        constexpr short mmNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(mmNumBanks));

        [[intel::numbanks(mmNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT a_load[B_rows][B_columns];
        fpga_tools::UnrolledLoop<B_rows>([&](auto r) {
            fpga_tools::UnrolledLoop<B_columns>([&](auto c) {
                if (r == c) a_load[r][c] = 1.0;
                else a_load[r][c] = 0.0;
            });
        });

        bool output_mm_result = false;
            
        for (int iteration = 0; iteration < iteration_count; iteration ++) {

            [[intel::numbanks(bNumBanksNextPow2)]]  // NO-FORMAT: Attribute
            [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
            [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
            [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
            TT b_load[B_rows][B_columns];

            [[intel::numbanks(mmNumBanksNextPow2)]]  // NO-FORMAT: Attribute
            [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
            [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
            [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
            TT mm_result[B_rows][B_columns];

            // Copy a matrix from the pipe to a local memory
            // Number of pipe reads of pipe_size required to read a full column
            constexpr int bExtraIteration = ((B_rows % pipe_size) != 0) ? 1 : 0;
            constexpr int mmExtraIteration = ((B_rows % pipe_size) != 0) ? 1 : 0;
            constexpr int bLoopIterPerColumn = (B_rows / pipe_size) + bExtraIteration;
            constexpr int mmLoopIterPerColumn = (B_rows / pipe_size) + mmExtraIteration;
            // Number of pipe reads of pipe_size to read all the matrices
            constexpr int bLoopIter = bLoopIterPerColumn * B_columns;
            constexpr int mmLoopIter = mmLoopIterPerColumn * B_columns;
            // Size in bits of the loop iterator over kLoopIter iterations
            constexpr int bLoopIterBitSize =
                fpga_tools::BitsForMaxValue<bLoopIter + 1>();
            constexpr int mmLoopIterBitSize =
                fpga_tools::BitsForMaxValue<mmLoopIter + 1>();

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
            for (int row = 0; row < B_rows; row++) {
                for (int column = 0; column < B_columns; column++) {
                    TT dot_prod{0};
                    fpga_tools::UnrolledLoop<B_columns>([&](auto k) {
                        // Assume dot_prods the B matrix was given transposed, otherwise it need to
                        // be transposed.
                        dot_prod = sycl::ext::intel::fpga_reg(dot_prod) +
                                a_load[row][k] * b_load[k][column];
                    });
                    mm_result[row][column] = dot_prod;
                }
            }

            // copy result matrix back to A
            fpga_tools::UnrolledLoop<B_rows>([&](auto r) {
                fpga_tools::UnrolledLoop<B_columns>([&](auto c) {
                    a_load[r][c] = mm_result[r][c];
                });
            });

            output_mm_result = OutEnable_pipe::read();
            if (output_mm_result) {
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
                            if constexpr (t * pipe_size + k < B_rows) {
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
            // IterationsFinished::write(iteration+1);
            // PRINTF("Matmul iteration %d done\n", iteration);
            }
        } // end of for loop (iterations)
    }    // end of operator
};     // end of structice

} // end of fpga_svd

#endif //__ACCUMULATOR_B_HPP__