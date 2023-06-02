#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

/*
    A 2-to-1 multiplexer, selecting between 2 input pipes and
    forward to 1 output pipe.
    This mux by default forwards from Pipe1 until 
    at lease 1 true packet is received trough Pipe1Done
*/
template <  typename TT,
            typename Pipe1Done,
            typename InPipe0,
            typename InPipe1,
            typename OutPipe>
struct PipeMux2to1
{
    void operator()() const
    {   
        // toggle state.
        int pipe1_iterations = 0;
        while (true) 
        {
            bool read_success;

            // update the pipe1 state
            int p1_done_read = Pipe1Done::read(read_success);
            if (read_success) pipe1_iterations = p1_done_read;

            if (pipe1_iterations >= 1) {
                // keep read until nothing left in the pipe buffer
                do {
                    auto data_read = InPipe0::read(read_success);
                    if (read_success) OutPipe::write(data_read);
                } while (read_success);
            }
            else {
                do {
                    auto data_read = InPipe1::read(read_success);
                    if (read_success) OutPipe::write(data_read);
                } while (read_success);
            }
        }
    }
    
};   // end of struct PipeMux2to1


template <  typename TT,
            typename InPipe,
            typename OutPipe0,
            typename OutPipe1>
struct PipeDuplicator2x
{
    void operator()() const
    {
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        while (true)
        {
            bool read_success;
            auto data_read = InPipe::read(read_success);
            if (read_success) {
                OutPipe0::write(data_read);
                OutPipe1::write(data_read);
            }
        }
    }
};