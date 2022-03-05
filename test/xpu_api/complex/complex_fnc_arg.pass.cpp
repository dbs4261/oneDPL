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

#include "support/utils_complex_test_support.h"

#include "complex_fnc_abs_cases.h"

using TestUtils::Complex::check_type;

////////////////////////////////////////////////////////////////////////////////
// class TestComplexArg - testing of std::arg from <complex>
// 
// Function std::conj described https://en.cppreference.com/w/cpp/numeric/complex/arg :
// 
//      template< class T >
//      T arg(const complex<T>& z);
//
//      since C++11
//          long double arg(long double z);
//          template< class DoubleOrInteger >
//          double arg(DoubleOrInteger z);
//          float arg(float z);
template <typename TErrorEngine, typename IsSupportedDouble, typename IsSupportedLongDouble>
class TestComplexArg
{
public:

    TestComplexArg(TErrorEngine& ee)
        : errorEngine(ee)
    {
    }

    template <typename RunOnHost>
    void run_test(RunOnHost /*runOnHost*/)
    {
        test_arg<float>();

        // Sometimes device, on which SYCL::queue work, may not support double type
        TestUtils::invoke_test_if<IsSupportedDouble>()([&](){ test_arg<double>(); });

        // Type "long double" not specified in https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#table.types.fundamental
        TestUtils::invoke_test_if<IsSupportedLongDouble>()([&](){ test_arg<long double>(); });

        TestUtils::invoke_test_if<IsSupportedDouble>()([&]() { test_arg_edges(); });

        // long double arg(long double z);
        TestUtils::invoke_test_if<IsSupportedLongDouble>()(
            [&]()
            {
                long double ldVal = 1.2;
                test_arg_for_non_complex_arg(ldVal);
            });

        // template< class DoubleOrInteger >
        // double arg(DoubleOrInteger z);
        TestUtils::invoke_test_if<IsSupportedDouble>()(
            [&]()
            {
                double dVal = 1.2;                      // DoubleOrInteger, result type checked
                test_arg_for_non_complex_arg(dVal);
            });
        {
            int iVal = 1;                               // DoubleOrInteger, result type checked
            test_arg_for_non_complex_arg(iVal);
        }

        // float arg(float z);
        {
            float fVal = 1.2f;
            test_arg_for_non_complex_arg(fVal);
        }
    }

protected:

    template <typename T>
    void test_arg()
    {
        const auto cv = dpl::complex<T>(TestUtils::Complex::InitConst<T>::kPartReal, TestUtils::Complex::InitConst<T>::kPartImag);
        const auto arg_res = dpl::arg(cv);
        check_type<T>(arg_res);
        auto arg_res_expected = ::std::arg(cv);

        EXPECT_TRUE_EE(errorEngine, arg_res == arg_res_expected, "Wrong result in dpl::arg(dpl::complex<T>()) function");
    }

    template <typename T>
    void test_arg_for_non_complex_arg(T val)
    {
        const auto arg_res = dpl::arg(val);
        TestUtils::Complex::check_type<typename TestUtils::Complex::InitConst<T>::DestComplexFieldType>(arg_res);

        const auto arg_res_expected = ::std::arg(val);

        EXPECT_TRUE_EE(errorEngine, arg_res == arg_res_expected, "Wrong result in dpl::arg(dpl::complex<T>()) function #1");
    }

    void test_arg_edges()
    {
        const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
        for (unsigned i = 0; i < N; ++i)
        {
            const auto cv = testcases[i];
            const double arg_res = dpl::arg(cv);
            check_type<double>(arg_res);
            const auto arg_res_expected = ::std::arg(cv);

            EXPECT_TRUE_EE(errorEngine, arg_res == arg_res_expected, "Wrong result in dpl::arg(dpl::complex<T>()) function #2");
        }
    }

private:

    TErrorEngine& errorEngine;
};

int
main()
{
    TestUtils::Complex::test_on_host<TestComplexArg>();

#if TEST_DPCPP_BACKEND_PRESENT
    try
    {
        sycl::queue deviceQueue{ TestUtils::default_selector };

        TestUtils::Complex::test_in_kernel<TestComplexArg>(deviceQueue);
    }
    catch (const std::exception& exc)
    {
        std::string errorMsg = "Exception occurred";
        if (exc.what())
        {
            errorMsg += " : ";
            errorMsg += exc.what();
        }

        EXPECT_TRUE(false, errorMsg.c_str());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
