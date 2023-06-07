#include <iostream>
#include <vector>
#include <chrono>
#include <type_traits>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

// #define LARGE_MATRIX
#define EPSILON 2E-6
#define MAX_CONVERGENCY_ERROR 0.05

#include "svd_helper.hpp"
#include "svd.hpp"
#include "svd_testcase.hpp"




int main()
{
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    // create the device queue
    sycl::queue q(selector);
    // make sure the device supports USM host allocations
    auto device = q.get_device();
    std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>().c_str()
            << std::endl;
    // if (!device.has(sycl::aspect::usm_host_allocations)) {
    //     std::cout << "Device dose not support USM, stop!" << std::endl;
    //     std::terminate();
    // }

    auto start = std::chrono::high_resolution_clock::now();
    // double delta = small_6x5_trivial.run_test(q, true);
    double delta = small_4x4_trivial.run_test(q, true);
    auto end = std::chrono::high_resolution_clock::now();
    // small_6x5_trivial.print_result();
    small_4x4_trivial.print_result();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Kernel runtime: "
         << duration.count() << " milliseconds" << std::endl;
    if (delta < 0.01) std::cout << "PASSED" << std::endl;

}