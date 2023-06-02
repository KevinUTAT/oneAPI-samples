#ifndef _ORTHOGONALIZER_HPP_
#define _ORTHOGONALIZER_HPP_

#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"

namespace fpga_svd {
/*
* Orthogonazing a matrix.
* It is a strip down version of QRD
*/

template <  typename TT,
            bool is_complex,
            int rows,
            int cols,
            int raw_latency,
            short banks,
            short bankw>
void orthogonazer_func(TT in_mat[rows][cols])
{
    [[intel::numbanks(banks)]]  // NO-FORMAT: Attribute
    [[intel::bankwidth(bankw)]]        // NO-FORMAT: Attribute
    [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
    TT out_mat[rows][cols];
    // Main loop of MGS.
    for (int i = 0; i < cols; i++)
    {
        TT t_magnitude_inv = 1.0f;
        // find magnitude of t_(*i) (i-th column)
        TT sum = 0;
#pragma unroll
        for (int row = 0; row < rows; row++)
        {
            TT val = in_mat[row][i];
            if constexpr (is_complex) sum = sum + val.mag_sqr();
            else sum = sum + (val * val);
        }
        //line 2
        t_magnitude_inv = 1.0/sqrt(sum);

        // line 3
        // r_matrix[i][i] = 1.0f / t_magnitude_inv;

        // line 4 of algorithm
        // generate the ith column of the Q matrix
#pragma unroll
        for (int mRow = 0; mRow < rows; mRow++)
        {
            out_mat[mRow][i] = in_mat[mRow][i] * t_magnitude_inv;
        }

        [[intel::ivdep]]
        for (int j = i + 1; j < cols; j++)
        {
            // line 5, 6 of algorithm
            TT dotProduct = 0;
#pragma unroll
            for (int mRow = 0; mRow < rows; mRow++)
            {
                // complex inner product <q_i, t_j>
                if constexpr (is_complex) dotProduct += out_mat[mRow][i]	* in_mat[mRow][j].conj();
                else dotProduct += out_mat[mRow][i]	* in_mat[mRow][j];
            }
            // PRINTF("%.2f+%.2fj\n", dotProduct.r(), dotProduct.i());
            // r_matrix[i][j] = dotProduct;
            // r_matrix[j][i] = 0.0f;

            // line 7 of algorithm
            // generate the next column of the T-matrix.
#pragma unroll
            for (int mRow = 0; mRow < rows; mRow++)
            {
                TT tj = in_mat[mRow][j];
                TT qi = out_mat[mRow][i];
                if constexpr (is_complex) in_mat[mRow][j] = tj - (dotProduct.conj() * qi);
                else in_mat[mRow][j] = tj - (dotProduct * qi);
            }
        }
    }

    // update the original matrix 
    fpga_tools::UnrolledLoop<rows>([&](auto r) {
        fpga_tools::UnrolledLoop<cols>([&](auto c) {
            in_mat[r][c] = out_mat[r][c];
        });
    });
}

} // end of namespace fpga_svd

#endif // _ORTHOGONALIZER_HPP_