#ifndef __IS_UPPER_TRIANGULAR_HPP__ 
#define __IS_UPPER_TRIANGULAR_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_svd {

/*
    A inline streaming kernal
    Check the diagonal elements of a matrix for convergency.
    If converged, out put to a different pipe
*/

template <  typename T,        // data type of incoming stream
            bool is_complex,    // if incoming data is ac_complex
            int rows,
            int cols,
            int pipe_size,
            typename In_pipe,   // incoming data
            typename Out_pipe,  // pass through output
            typename Final_pipe,// pipe for the final output
            typename Converge_pipe>
struct DiagonalConvergence {
    int max_iteration;
    float epsilon;  // small number that can be considered zero
    float max_error;   // max interation to interation error to be considered converged
    void operator()() const {
        using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
        bool converged = false;
        constexpr int diagonal_size = (rows > cols) ? cols : rows; // min(rows, cols)
        TT diagonals[diagonal_size] = {(TT)0.0};

        for (int iteration = 0; iteration < max_iteration; iteration ++) {
            bool converge_sofar = true;
            if (iteration <= 1) converge_sofar = false;   // at least 2 iterations are needed to check for convergency
            // Copy a matrix from the pipe to a local memory
            // Number of pipe reads of pipe_size required to read a full column
            constexpr int iExtraIteration = ((rows % pipe_size) != 0) ? 1 : 0;
            constexpr int iLoopIterPerColumn = (rows / pipe_size) + iExtraIteration;
            // Number of pipe reads of pipe_size to read all the matrices
            constexpr int iLoopIter = iLoopIterPerColumn * cols;
            // Size in bits of the loop iterator over kLoopIter iterations
            constexpr int iLoopIterBitSize =
                fpga_tools::BitsForMaxValue<iLoopIter + 1>();
            
            // read input matrix from pipe
            [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
            for (ac_int<iLoopIterBitSize, false> li = 0; li < iLoopIter; li++) {
                fpga_tools::NTuple<TT, pipe_size> pipe_read_in = In_pipe::read();
                

                if (!converged) {
                    // immediately pass the data down
                    Out_pipe::write(pipe_read_in);

                    int write_idx = li % iLoopIterPerColumn;
                    fpga_tools::UnrolledLoop<iLoopIterPerColumn>([&](auto k) {
                        fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
                            
                            if constexpr (k * pipe_size + t < rows) {
                                if (write_idx == k) {
                                    TT cur_element = pipe_read_in.template get<t>();
                                    int cur_row = k * pipe_size + t;
                                    int cur_col = li / iLoopIterPerColumn;
                                    // if is diagonal
                                    if (cur_row == cur_col) {
                                        // except the case where element converge to zero
                                        if ((cur_element > epsilon) && (diagonals[cur_row] > epsilon)) {
                                            float diff = fabs((float)(cur_element - diagonals[cur_row]));
                                            float error = diff / fabs(cur_element);   // error could be -+ inf or NaN
                                            if (error > max_error) {
                                                converge_sofar = false;
                                                // PRINTF("Not converged: [%d][%d]=%.4f, last iteration: %.4f\n", 
                                                //         cur_row, cur_col, cur_element, diagonals[cur_row]);
                                            }
                                        }
                                        diagonals[cur_row] = cur_element;
                                        // if (fabs((float)(cur_element - diagonals[cur _row])) > epsilon) {
                                        //     converge_sofar = false;
                                        //     PRINTF("Not converged: [%d][%d]=%.4f, last iteration: %.4f\n", 
                                        //             cur_row, cur_col, cur_element, diagonals[cur_row]);
                                        //     diagonals[cur_row] = cur_element;
                                        // }
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
                else {
                    Final_pipe::write(pipe_read_in);
                }
            }
            // PRINTF("Convergency: %d\n", converge_sofar);
            // if its the second last iteration, mark as "converged" no mater what
            if (iteration >= (max_iteration - 2)) converged = true;
            else converged = converge_sofar;

            // output current convergency status
            Converge_pipe::write(converged);
        }
    }
};

}  // namespace fpga_svd

#endif  // __IS_UPPER_TRIANGULAR_HPP__