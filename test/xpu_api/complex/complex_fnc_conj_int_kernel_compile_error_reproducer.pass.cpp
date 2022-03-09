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
                cgh.single_task<class TestConjCompile>(
                    [=]()
                    {
                        int iv = 1;
                        auto cv_conj_res = std::conj(iv);
                        ::std::complex<double> cv(cv_conj_res.real(), cv_conj_res.imag());
                        cv = cv;
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
