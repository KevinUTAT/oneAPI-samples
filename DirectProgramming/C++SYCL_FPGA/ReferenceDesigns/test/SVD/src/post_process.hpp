#ifndef _POST_PROCESS_HPP_
#define _POST_PROCESS_HPP_

#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"

namespace fpga_svd {

/*
* Building S matrix from accumilated R matrix.
* eigen value as diagnal and resize 
*/
template <  typename T,
            bool is_complex,
            int R_rows,
            int R_cols,
            int S_rows,
            int S_cols,
            int pipe_size,
            typename RIn,
            typename SOut>
struct SBuilder {
    int iteration_count;
    void operator()() const {
        using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
        // constexpr int diagonal_size = (R_rows > R_cols) ? R_cols : R_rows; // min(rows, cols)
        // TT diagonals[diagonal_size] = {(TT)0.0};

        for (int iteration = 0; iteration < iteration_count; iteration ++) {
            // read in R and extract the diagnal elements
            constexpr int diagonal_size = (S_rows > S_cols) ? S_cols : S_rows; // min(rows, cols)
            // Copy a matrix from the pipe to a local memory
            // Number of pipe reads of pipe_size required to read a full column
            constexpr int iExtraIteration = ((R_rows % pipe_size) != 0) ? 1 : 0;
            constexpr int sExtraIteration = ((S_rows % pipe_size) != 0) ? 1 : 0;
            constexpr int iLoopIterPerColumn = (R_rows / pipe_size) + iExtraIteration;
            constexpr int sLoopIterPerColumn = (S_rows / pipe_size) + sExtraIteration;
            // Number of pipe reads of pipe_size to read all the matrices
            constexpr int iLoopIter = iLoopIterPerColumn * R_cols;
            constexpr int sLoopIter = sLoopIterPerColumn * R_cols;
            // Size in bits of the loop iterator over kLoopIter iterations
            constexpr int iLoopIterBitSize =
                fpga_tools::BitsForMaxValue<iLoopIter + 1>();
            constexpr int sLoopIterBitSize =
                    fpga_tools::BitsForMaxValue<sLoopIter + 1>();
            
            constexpr short kBankwidth = pipe_size * sizeof(TT);
            constexpr unsigned short sNumBanks = S_rows / pipe_size;
            constexpr unsigned short iNumBanks = R_rows / pipe_size;
            constexpr short iNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(iNumBanks));
            constexpr short sNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(sNumBanks));

            [[intel::numbanks(iNumBanksNextPow2)]]  // NO-FORMAT: Attribute
            [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
            [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
            [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
            TT R_load[R_rows][R_cols];

            [[intel::numbanks(sNumBanksNextPow2)]]  // NO-FORMAT: Attribute
            [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
            [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
            [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
            TT S_result[S_rows][S_cols] = {};
            
            // read input matrix from pipe
#ifndef LARGE_MATRIX
            [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
            for (ac_int<iLoopIterBitSize, false> li = 0; li < iLoopIter; li++) {
                fpga_tools::NTuple<TT, pipe_size> pipe_read_in = RIn::read();
                
                int write_idx = li % iLoopIterPerColumn;
                fpga_tools::UnrolledLoop<iLoopIterPerColumn>([&](auto k) {
                    fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                        if constexpr (k * pipe_size + t < R_rows) {
                            if (write_idx == k) {
                                R_load[k * pipe_size + t][li / iLoopIterPerColumn] =
                                    pipe_read_in.template get<t>();
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

            fpga_tools::UnrolledLoop<diagonal_size>([&](auto d) {
                S_result[d][d] = sycl::sqrt(R_load[d][d]);
            });

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
                        if constexpr (t * pipe_size + k < S_rows) {
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

        } // end iteration
    } // operater ()
}; // struct

/*
*   Building the U matix knowing A, S, V
*   Using the relationship defined by SVD: A = U @ S @ Vt
*/
template<   typename T,
            bool is_complex,
            int AV_rows,
            int AV_cols,
            int S_rows,
            int S_cols,
            int pipe_size,
            typename AVIn_pipe, 
            typename SIn_pipe,
            typename UOut_pipe 
            >

struct UBuilder {
    void operator()() const {
        static_assert((AV_rows == S_rows) && (AV_cols == S_cols),
                    "Dimention of A@V and S should match");
        // PRINTF("Start building U\n");

        /*
        *   By A = U @ S @ Vt where S is diaginal, V is orthornomal 
        *   Therefor (A @ V)S^-1 = U, or :
        *   U[j][i] = AV[j][i] / S[i][i]
        */

        using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

        constexpr short kBankwidth = pipe_size * sizeof(TT);
        constexpr unsigned short avNumBanks = AV_rows / pipe_size;
        constexpr unsigned short sNumBanks = S_rows / pipe_size;
        constexpr unsigned short uNumBanks = S_rows / pipe_size;

        // When specifying numbanks for a memory, it must be a power of 2.
        // Unused banks will be automatically optimized away.
        constexpr short avNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(avNumBanks));
        constexpr short sNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(sNumBanks));
        constexpr short uNumBanksNextPow2 =
                fpga_tools::Pow2(fpga_tools::CeilLog2(uNumBanks));

        [[intel::numbanks(avNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT av_load[AV_rows][AV_cols];

        [[intel::numbanks(sNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT s_load[S_rows][S_cols];

        [[intel::numbanks(uNumBanksNextPow2)]]  // NO-FORMAT: Attribute
        [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT u_result[S_rows][S_rows];

        // Copy a matrix from the pipe to a local memory
        // Number of pipe reads of pipe_size required to read a full column
        constexpr int avExtraIteration = ((AV_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int sExtraIteration = ((S_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int uExtraIteration = ((S_rows % pipe_size) != 0) ? 1 : 0;
        constexpr int avLoopIterPerColumn = (AV_rows / pipe_size) + avExtraIteration;
        constexpr int sLoopIterPerColumn = (S_rows / pipe_size) + sExtraIteration;
        constexpr int uLoopIterPerColumn = (S_rows / pipe_size) + uExtraIteration;
        // Number of pipe reads of pipe_size to read all the matrices
        constexpr int avLoopIter = avLoopIterPerColumn * AV_cols;
        constexpr int sLoopIter = sLoopIterPerColumn * S_cols;
        constexpr int uLoopIter = uLoopIterPerColumn * S_rows;
        // Size in bits of the loop iterator over kLoopIter iterations
        constexpr int avLoopIterBitSize =
            fpga_tools::BitsForMaxValue<avLoopIter + 1>();
        constexpr int sLoopIterBitSize =
            fpga_tools::BitsForMaxValue<sLoopIter + 1>();
        constexpr int uLoopIterBitSize =
            fpga_tools::BitsForMaxValue<uLoopIter + 1>();

        // load matrix AV from pipe
#ifndef LARGE_MATRIX
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
        for (ac_int<avLoopIterBitSize, false> li = 0; li < avLoopIter; li++) {
            fpga_tools::NTuple<TT, pipe_size> pipe_read_av = AVIn_pipe::read();

            int write_idx_av = li % avLoopIterPerColumn;
            fpga_tools::UnrolledLoop<avLoopIterPerColumn>([&](auto k) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                    
                    if constexpr (k * pipe_size + t < AV_rows) {
                        if (write_idx_av == k) {
                            av_load[k * pipe_size + t][li / avLoopIterPerColumn] =
                                pipe_read_av.template get<t>();
                        }
                    }

                    // Delay data signals to create a vine-based data distribution
                    // to lower signal fanout.
                    pipe_read_av.template get<t>() =
                        sycl::ext::intel::fpga_reg(pipe_read_av.template get<t>());
                });
            
                write_idx_av = sycl::ext::intel::fpga_reg(write_idx_av);
            });
        }

        // load matrix S from pipe
#ifndef LARGE_MATRIX
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
#endif
        for (ac_int<sLoopIterBitSize, false> li = 0; li < sLoopIter; li++) {
            fpga_tools::NTuple<TT, pipe_size> pipe_read_s = SIn_pipe::read();

            int write_idx_s = li % sLoopIterPerColumn;

            fpga_tools::UnrolledLoop<sLoopIterPerColumn>([&](auto k) {
                fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {

                    if constexpr (k * pipe_size + t < S_rows) {
                        if (write_idx_s == k) {
                            s_load[k * pipe_size + t][li / sLoopIterPerColumn] =
                                pipe_read_s.template get<t>();
                        }
                    }

                    // Delay data signals to create a vine-based data distribution
                    // to lower signal fanout.
                    pipe_read_s.template get<t>() =
                        sycl::ext::intel::fpga_reg(pipe_read_s.template get<t>());
                });
                
                write_idx_s = sycl::ext::intel::fpga_reg(write_idx_s);
            });
        }

        // fill matrix U
        int diagonal_size = (S_rows > S_cols) ? S_cols : S_rows; // min(S_rows, S_cols)
        fpga_tools::UnrolledLoop<S_rows>([&](auto r) {
            fpga_tools::UnrolledLoop<S_rows>([&](auto c) {
                if (c < diagonal_size) {
                    TT s_val = s_load[c][c];
                    if (s_val > 2e-20) {
                        u_result[r][c] = av_load[r][c] / s_val;
                    }
                    else u_result[r][c] = 0.1;  // place holder value
                }
                else u_result[r][c] = 0.1;
            });
        });

        // // debug ========================
        // for (int row = 0; row < S_rows; row ++)
        // {
        //     for (int col = 0; col < S_rows; col ++)
        //     {
        //         // PRINTF("%.2f ", mm_result[row][col]);
        //         PRINTF("%.2f ", u_result[row][col]);
        //     }
        //     PRINTF("\n");
        // }
        // PRINTF("---------------------\n");
        // // ==============================

        // Copy the result U matrix to the output pipe (in col maj)
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
                    if constexpr (t * pipe_size + k < S_rows) {
                        pipe_write.template get<k>() =
                            get[t] ? u_result[t * pipe_size + k][li / uLoopIterPerColumn]
                                    : sycl::ext::intel::fpga_reg(
                                        pipe_write.template get<k>());
                        // PRINTF("%.2f ", pipe_write.template get<k>());
                    }
                });
            });
            UOut_pipe::write(pipe_write);
        }
    }
};


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
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
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