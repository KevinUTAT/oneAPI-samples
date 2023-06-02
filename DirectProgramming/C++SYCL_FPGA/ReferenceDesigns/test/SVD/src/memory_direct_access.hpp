#ifndef __MEMORY_DIRECT_ACCESS_HPP__
#define __MEMORY_DIRECT_ACCESS_HPP__
/*
*   Memory accesses for matrices. Based on memory_transfers.hpp from 
*   QRD and QRI reference design
*/

#include "tuple.hpp"
#include "constexpr_math.hpp"
#include "unrolled_loop.hpp"
#include "dpc_common.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          typename MatrixPipe    // Output matrix pipe
          >
void MatrixReadFromDDRToPipeColMaj(
    TT* matrix_ptr  // Input matrix pointer
    ) {

    // We may perform an incomplete memory read if the number of elements per row
    // is not a multiple of the DDR burst size
    constexpr bool kIncompleteBurst = rows%num_elem_per_bank != 0;
    constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
    // Number of DDR burst reads of num_elem_per_bank elements required to read a
    // full column
    constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
    // Number of DDR burst reads of num_elem_per_bank to read all the matrices
    constexpr int kLoopIter = kLoopIterPerColumn * columns;
    // Size in bits of the loop iterator over kLoopIter iterations
    constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

    // sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

    // Keep track of the current element index in the matrix
    // Only useful in the case of kIncompleteBurst
    int load_index = 0;

    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst){
            // Check if we are reading the last DDR burst of the current column
            last_burst_of_col = (li % kLoopIterPerColumn)
                                == kLoopIterPerColumn - 1;
        }

        fpga_tools::NTuple<TT, num_elem_per_bank> ddr_read;

        // Perform the DDR burst read of num_elem_per_bank elements
        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {

            if constexpr (kIncompleteBurst){
                // Check if the current read index is beyond the end of the current
                // matrix column
                bool out_of_bounds = last_burst_of_col &&
                        ((k % num_elem_per_bank) > ((rows - 1) % num_elem_per_bank));

                // Only perform the DDR reads that are relevant (and don't access a
                // memory address that may be beyond the matrix last address)
                if (!out_of_bounds) {
                    ddr_read.template get<k>() = matrix_ptr
                                        [load_index + k];
                }
            }
            else{
                ddr_read.template get<k>() = matrix_ptr
                    [(int)(li)*num_elem_per_bank + k];

            }
        });

        if constexpr (kIncompleteBurst){
            // Update the current element index in the input matrix according
            // to the read size of the current iteration
            load_index += last_burst_of_col ? rows % num_elem_per_bank :
                                            num_elem_per_bank;
        }

        MatrixPipe::write(ddr_read);
    }  // end of li
    // PRINTF("Pipe write done.\n");
}


template <typename TT,           // Datatype of the elements of the matrix
          int rows,              // Number of rows of the matrix
          int columns,           // Number of columns of the matrix
          int num_elem_per_bank, // Number of TT elements per DDR burst access
          typename MatrixPipe    // Input matrix
          >
void MatrixReadPipeToDDRColMaj(
    TT* matrix_ptr,  // Output matrix pointer
    int iteration_count = 1 // How many matrices are expected be for kernel returns
    ) {

    // We may perform an incomplete memory write if the number of elements per row
    // is not a multiple of the DDR burst size
    constexpr bool kIncompleteBurst = rows%num_elem_per_bank != 0;
    constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
    // Number of DDR burst of num_elem_per_bank required to write a full column
    constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
    // Number of DDR burst of num_elem_per_bank to write all the matrices
    constexpr int kLoopIter = kLoopIterPerColumn * columns;
    // Size in bits of the loop iterator over kLoopIter iterations
    constexpr int kLoopIterBitSize = fpga_tools::BitsForMaxValue<kLoopIter + 1>();

    // sycl::device_ptr<TT> matrix_ptr_device(matrix_ptr);

    for (int iteration = 0; iteration < iteration_count; iteration ++) {
        // Keep track of the current element index in the output matrix
        // Only useful in the case of kIncompleteBurst
        int write_idx = 0;

        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        [[intel::ivdep]]  // NO-FORMAT: Attribute
        for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
            fpga_tools::NTuple<TT, num_elem_per_bank> pipe_read =
                                                            MatrixPipe::read();

            bool last_burst_of_col;
            if constexpr (kIncompleteBurst){
                // Check if we are writing the last DDR burst of the current column
                last_burst_of_col =
                            (li % kLoopIterPerColumn) == kLoopIterPerColumn - 1;
            }

            fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
                if constexpr (kIncompleteBurst){
                    // Check if the current write index is beyond the end of the current
                    // matrix column
                    bool out_of_bounds = last_burst_of_col &&
                                            (k > ((rows - 1) % num_elem_per_bank));
                    // Only perform the DDR writes that are relevant (and don't access a
                    // memory address that may be beyond the buffer last address)
                    if (!out_of_bounds) {
                        matrix_ptr[write_idx + k] =
                                                            pipe_read.template get<k>();
                    }
                }
                else{
                    matrix_ptr[int(li) * num_elem_per_bank + k] 
                            = pipe_read.template get<k>();
                }

            });

            if constexpr (kIncompleteBurst){
                // Update the current element index in the write buffer according
                // to the write size of the current iteration
                write_idx += last_burst_of_col ? rows % num_elem_per_bank :
                                                num_elem_per_bank;
            }
        }  // end of li
    }   // end of iteration
    // PRINTF("DDR write done.\n");
}


/*
    Collect and update value coming from a pipe
*/
template   <typename TT,        // the type of the pipe
            typename InPipe>    // Incoming pipe
struct CollectPipeToDDR
{
    TT* current_value;

    void operator()() const
    {
        while (true)
        {
            TT read_val = InPipe::read();   // blocking read
            current_value[0] = read_val;
        }
    }
};


#endif /* __MEMORY_DIRECT_ACCESS_HPP__ */