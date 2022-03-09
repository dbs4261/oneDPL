// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <complex>
#include <exception>
#include <iostream>

#include <CL/sycl.hpp>

int
main()
{
    try
    {
        sycl::queue deviceQueue;

        deviceQueue.submit(
            [&](cl::sycl::handler& cgh)
            {
                cgh.single_task<class TestAbsCompile>(
                    [=]()
                    {
                        auto cv = ::std::complex<float>(1.5f, 2.25f);
                        auto abs_res = ::std::abs(cv);
                        abs_res = abs_res;
                    });
            });

    }
    catch (const std::exception& exc)
    {
        std::string errorMsg = "Exception occurred";
        if (exc.what())
        {
            errorMsg += " : ";
            errorMsg += exc.what();
        }

        std::cout << errorMsg << std::endl;
    }

    return 0;
}
