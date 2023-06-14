#ifndef _POST_PROCESS_HPP_
#define _POST_PROCESS_HPP_

#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"

namespace fpga_svd {

template <  typename T,
            bool is_complex,
            int A_rows,
            int A_cols,
            int pipe_size,
            typename AIn,   // Original SVD input A,        size A_rows x A_cols
            typename RIn,   // R matrix from QR iteration,  size A_cols x A_cols
            typename VIn,   // Q accumilated V input,       size A_cols x A_cols
            typename UOut,  // U output                     size A_rows x A_rows
            typename SOut,  // S output                     size A_rows x A_cols
            typename VOut>  // V output                     size A_cols x A_cols
struct PostProcess {
    void operator()() const {
        // PRINTF("Start post processing\n");
        using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
        // findout if orthogonalazation is needed on the output
        constexpr bool U_not_orthogonal = A_rows > A_cols;
        constexpr bool V_not_orthogonal = A_rows < A_cols;

        constexpr int diagonal_size = (A_rows > A_cols) ? A_cols : A_rows; // min(rows, cols)
        // Copy a matrix from the pipe to a local memory
        // Number of pipe reads of pipe_size required to read a full column
        constexpr int rExtraIteration = ((A_cols % pipe_size) != 0) ? 1 : 0;
        constexpr int sExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int aExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int vExtraIteration = ((A_cols % pipe_size) != 0) ? 1 : 0;
        constexpr int uExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int rLoopIterPerColumn = (A_cols / pipe_size) + rExtraIteration;
        constexpr int sLoopIterPerColumn = (A_rows / pipe_size) + sExtraIteration;
        constexpr int aLoopIterPerColumn = (A_rows / pipe_size) + aExtraIteration;
        constexpr int vLoopIterPerColumn = (A_cols / pipe_size) + vExtraIteration;
        constexpr int uLoopIterPerColumn = (A_rows / pipe_size) + uExtraIteration;
        // Number of pipe reads of pipe_size to read all the matrices
        constexpr int rLoopIter = rLoopIterPerColumn * A_cols;
        constexpr int sLoopIter = aLoopIterPerColumn * A_cols;
        constexpr int aLoopIter = aLoopIterPerColumn * A_cols;
        constexpr int vLoopIter = vLoopIterPerColumn * A_cols;
        constexpr int uLoopIter = uLoopIterPerColumn * A_rows;
        // Size in bits of the loop iterator over kLoopIter iterations
        constexpr int rLoopIterBitSize =
            fpga_tools::BitsForMaxValue<rLoopIter + 1>();
        constexpr int sLoopIterBitSize =
            fpga_tools::BitsForMaxValue<sLoopIter + 1>();
        constexpr int aLoopIterBitSize =
            fpga_tools::BitsForMaxValue<aLoopIter + 1>();
        constexpr int vLoopIterBitSize =
            fpga_tools::BitsForMaxValue<rLoopIter + 1>();
        constexpr int uLoopIterBitSize =
            fpga_tools::BitsForMaxValue<uLoopIter + 1>();
        
        constexpr short kBankwidth = pipe_size * sizeof(TT);
        constexpr unsigned short sNumBanks = A_rows / pipe_size;
        constexpr unsigned short aNumBanks = A_rows / pipe_size;
        constexpr unsigned short vNumBanks = A_rows / pipe_size;
        constexpr unsigned short uNumBanks = A_rows / pipe_size;
        constexpr short sNumBanksNextPow2 =
            fpga_tools::Pow2(fpga_tools::CeilLog2(sNumBanks));
        constexpr short aNumBanksNextPow2 =
            fpga_tools::Pow2(fpga_tools::CeilLog2(aNumBanks));
        constexpr short vNumBanksNextPow2 =
            fpga_tools::Pow2(fpga_tools::CeilLog2(vNumBanks));
        constexpr short uNumBanksNextPow2 =
            fpga_tools::Pow2(fpga_tools::CeilLog2(uNumBanks));


        [[intel::numbanks(sNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT S_result[A_rows][A_cols];

        [[intel::numbanks(aNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT A_load[A_rows][A_cols];

        [[intel::numbanks(vNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT V_load[A_cols][A_cols];

        [[intel::numbanks(uNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(8)]]            // NO-FORMAT: Attribute
        TT U_result[A_rows][A_rows];
        // PRINTF("Start load R\n");
        // load Rin into S_rusult, while resize and sqrt
#ifndef LARGE_MATRIX
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
        for (ac_int<rLoopIterBitSize, false> li = 0; li < rLoopIter; li++) {
            fpga_tools::NTuple<TT, pipe_size> pipe_read_in = RIn::read();
            int write_idx = li % rLoopIterPerColumn;
            fpga_tools::UnrolledLoop<rLoopIterPerColumn>([&](auto k) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                    if constexpr (k * pipe_size + t < A_cols) {
                        if (write_idx == k) {
                            int cur_row = k * pipe_size + t;
                            int cur_col = li / rLoopIterPerColumn;
                            if (cur_row < A_rows) {
                                if (cur_row == cur_col) {
                                    S_result[cur_row][cur_col] =  
                                            pipe_read_in.template get<t>();
                                }
                                // else S_result[cur_row][cur_col] = (TT)0.0;
                            }
                        }
                    }
                    // Delay data signals to create a vine-based data distribution
                    // to lower signal fanout.
                    pipe_read_in.template get<t>() =
                        sycl::ext::intel::fpga_reg(pipe_read_in.template get<t>());
                });
            
                write_idx = sycl::ext::intel::fpga_reg(write_idx);
            });
        }
        
        // process R (sqrt and zero pading)
        fpga_tools::UnrolledLoop<A_rows>([&](auto r) {
            fpga_tools::UnrolledLoop<A_cols>([&](auto c) {
                if (r == c) S_result[r][c] = sycl::sqrt(S_result[r][c]);
                else S_result[r][c] = (TT)0.0;
            });
        });
        // PRINTF("S_loaded\n");
        // load A
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
                            A_load[k * pipe_size + t][li / aLoopIterPerColumn] =
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
        // PRINTF("A loaded\n");
        // load V
#ifndef LARGE_MATRIX
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
        for (ac_int<vLoopIterBitSize, false> li = 0; li < vLoopIter; li++) {
            fpga_tools::NTuple<TT, pipe_size> pipe_read_v = VIn::read();

            int write_idx_v = li % vLoopIterPerColumn;
            fpga_tools::UnrolledLoop<vLoopIterPerColumn>([&](auto k) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                    
                    if constexpr (k * pipe_size + t < A_cols) {
                        if (write_idx_v == k) {
                            V_load[k * pipe_size + t][li / vLoopIterPerColumn] =
                                pipe_read_v.template get<t>();
                        }
                    }

                    // Delay data signals to create a vine-based data distribution
                    // to lower signal fanout.
                    pipe_read_v.template get<t>() =
                        sycl::ext::intel::fpga_reg(pipe_read_v.template get<t>());
                });
            
                write_idx_v = sycl::ext::intel::fpga_reg(write_idx_v);
            });
        }
        // PRINTF("V loaded\n")

        // orthogonalize V if needed (determined at compile time)
        if constexpr (V_not_orthogonal) {
            fpga_svd::orthogonazer_func
                <TT, is_complex, A_cols, A_cols, 110, vNumBanksNextPow2, kBankwidth>
                (V_load);
        }

        // Compute the matrix product A @ V / S[c][c]
        for (int row = 0; row < A_rows; row++) {
            for (int column = 0; column < A_rows; column++) {
                if (column < diagonal_size) {
                    TT dot_prod{0};
                    fpga_tools::UnrolledLoop<A_cols>([&](auto k) {
                        // Assume dot_prods the B matrix was given transposed, otherwise it need to
                        // be transposed.
                        dot_prod = sycl::ext::intel::fpga_reg(dot_prod) +
                                A_load[row][k] * V_load[k][column];
                    });
                
                    TT s_val = S_result[column][column];
                    U_result[row][column] = dot_prod / s_val;
                }
                else U_result[row][column] = (TT)0.1;
            }
        }

        // orthogonalize U if needed (determined at compile time)
        if constexpr (U_not_orthogonal) {
            fpga_svd::orthogonazer_func
                <TT, is_complex, A_rows, A_rows, 110, uNumBanksNextPow2, kBankwidth>
                (U_result);
        }

        // output S_result
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (ac_int<sLoopIterBitSize, false> li = 0; li < sLoopIter; li++) {
            int column_iter = li % sLoopIterPerColumn;
            bool get[sLoopIterPerColumn];
            fpga_tools::UnrolledLoop<sLoopIterPerColumn>([&](auto k) {
                get[k] = column_iter == k;
                column_iter = sycl::ext::intel::fpga_reg(column_iter);
            });
    
            fpga_tools::NTuple<TT, pipe_size> pipe_write;
            fpga_tools::UnrolledLoop<sLoopIterPerColumn>([&](auto t) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                    if constexpr (t * pipe_size + k < A_rows) {
                        pipe_write.template get<k>() =
                            get[t] ? S_result[t * pipe_size + k][li / sLoopIterPerColumn]
                                    : sycl::ext::intel::fpga_reg(
                                        pipe_write.template get<k>());
                        // PRINTF("%.2f ", pipe_write.template get<k>());
                    }
                });
            });
            SOut::write(pipe_write);
        }

        // output V_load as is
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (ac_int<vLoopIterBitSize, false> li = 0; li < vLoopIter; li++) {
            int column_iter = li % vLoopIterPerColumn;
            bool get[vLoopIterPerColumn];
            fpga_tools::UnrolledLoop<vLoopIterPerColumn>([&](auto k) {
            get[k] = column_iter == k;
            column_iter = sycl::ext::intel::fpga_reg(column_iter);
            });
    
            fpga_tools::NTuple<TT, pipe_size> pipe_write;
            fpga_tools::UnrolledLoop<vLoopIterPerColumn>([&](auto t) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                    if constexpr (t * pipe_size + k < A_cols) {
                        pipe_write.template get<k>() =
                            get[t] ? V_load[t * pipe_size + k][li / vLoopIterPerColumn]
                                    : sycl::ext::intel::fpga_reg(
                                        pipe_write.template get<k>());
                        // PRINTF("%.2f ", pipe_write.template get<k>());
                    }
                });
            });
            VOut::write(pipe_write);
        }

        // output U_result
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (ac_int<uLoopIterBitSize, false> li = 0; li < uLoopIter; li++) {
            int column_iter = li % uLoopIterPerColumn;
            bool get[uLoopIterPerColumn];
            fpga_tools::UnrolledLoop<uLoopIterPerColumn>([&](auto k) {
            get[k] = column_iter == k;
            column_iter = sycl::ext::intel::fpga_reg(column_iter);
            });
    
            fpga_tools::NTuple<TT, pipe_size> pipe_write;
            fpga_tools::UnrolledLoop<uLoopIterPerColumn>([&](auto t) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                    if constexpr (t * pipe_size + k < A_rows) {
                        pipe_write.template get<k>() =
                            get[t] ? U_result[t * pipe_size + k][li / uLoopIterPerColumn]
                                    : sycl::ext::intel::fpga_reg(
                                        pipe_write.template get<k>());
                        // PRINTF("%.2f ", pipe_write.template get<k>());
                    }
                });
            });
            UOut::write(pipe_write);
        }
    }
}; // struct PostProcess

}   // namespace fpga_svd

#endif // _POST_PROCESS_HPP_