// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp --------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <CL/sycl.hpp>

#include <oneapi/tbb/global_control.h>
int
main(int argc, char* argv[])
{
    int size = 10;

    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);

    auto sycl_asynchandler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const& e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& ex)
            {
                std::cout << "Caught asynchronous SYCL exception:" << std::endl
                          << ex.what() << ", SYCL code: " << ex.code() << std::endl;
            }
        }
    };

    sycl::device* sycl_device = new sycl::device(sycl::gpu_selector{});
    //sycl::device* sycl_device = new sycl::device(sycl::default_selector{});
    sycl::context sycl_ctxt = sycl::context(*sycl_device, sycl_asynchandler);
    sycl::queue q(sycl_ctxt, *sycl_device, sycl::property_list{sycl::property::queue::in_order{}});

    int* keys_in = (int*)sycl::malloc_shared(size * sizeof(int), q);
    int* keys_out = (int*)sycl::malloc_shared(size * sizeof(int), q);
    int* vals_in = (int*)sycl::malloc_shared(size * sizeof(int), q);
    int* vals_out = (int*)sycl::malloc_shared(size * sizeof(int), q);

    for (int i = 0; i < size; i++)
    {
        keys_in[i] = 0;
        vals_in[i] = 1;
        keys_out[i] = 1;
        vals_out[i] = 0;
    }
    /* Uncomment the line below to make two groups of keys */
    /* keys_in[size - 1] = 1; */

    std::cout << "keys_in:";
    for (int i = 0; i < size; i++)
        std::cout << " " << keys_in[i];
    std::cout << std::endl;

    std::cout << "vals_in:";
    for (int i = 0; i < size; i++)
        std::cout << " " << vals_in[i];
    std::cout << std::endl;

    std::cout << "Calling reduce_by_segment" << std::endl;

    auto new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q), keys_in,
                                                  keys_in + size, vals_in, keys_out, vals_out);
    int new_size = new_end.second - vals_out;

    std::cout << "new_size = " << new_size << std::endl;

    std::cout << "keys_out:";
    for (int i = 0; i < size; i++)
        std::cout << " " << keys_out[i];
    std::cout << std::endl;

    std::cout << "vals_out:";
    for (int i = 0; i < size; i++)
        std::cout << " " << vals_out[i];
    std::cout << std::endl;

    std::cout << "DONE" << std::endl;
    return 0;
}
