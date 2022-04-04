// -*- C++ -*-
//===-- sycl_iterator.pass.cpp --------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
#include "oneapi/dpl/pstl/utils.h"

#include <cmath>
#include <type_traits>

using namespace TestUtils;

//This macro is required for the tests to work correctly in CI with tbb-backend.
#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"

using namespace oneapi::dpl::execution;

DEFINE_TEST(test_any_all_none_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_any_all_none_of)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1,
                                      [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(!res0, "wrong effect from any_of_0");
            res0 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1,
                                  [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0, "wrong effect from none_of_0");
            res0 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, first1,
                                 [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0, "wrong effect from all_of_0");
        }
        // any_of
        auto res1 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1,
                                  [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1, "wrong effect from any_of_1");
        auto res2 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1,
                                  [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(!res2, "wrong effect from any_of_2");
        auto res3 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1,
                                  [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res3, "wrong effect from any_of_3");

        //none_of
        auto res4 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1,
                                   [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res4, "wrong effect from none_of");

        //all_of
        auto res5 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1,
                                  [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(n == 1 || !res5, "wrong effect from all_of");
    }
};

DEFINE_TEST(test_equal)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = T(42);

        auto new_start = n / 3;
        auto new_end = n / 2;

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, T{0});
        ::std::fill(host_vals.get() + new_start, host_vals.get() + new_end, value);
        update_data(host_keys, host_vals);

        auto expected  = new_end - new_start > 0;
        auto result = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + new_start,
                                   first1 + new_end, first2 + new_start);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(expected == result, "wrong effect from equal with 3 iterators");
        result = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1 + new_start, first1 + new_end,
                              first2 + new_start, first2 + new_end);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(expected == result, "wrong effect from equal with 4 iterators");
    }
};

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1,
                                       [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from find_if_0");
            res0 = ::std::find(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, T1(1));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from find_0");
        }
        // find_if
        auto res1 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1,
                                   [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE((res1 - first1) == n - 1, "wrong effect from find_if_1");

        auto res2 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1,
                                   [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res2 == last1, "wrong effect from find_if_2");

        auto res3 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1,
                                   [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res3 == first1, "wrong effect from find_if_3");

        //find
        auto res4 = ::std::find(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, T1(-1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res4 == last1, "wrong effect from find");
    }
};

DEFINE_TEST(test_find_first_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_first_of)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        // Reset values after previous execution
        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        if (n < 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto res =
                ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first1, "Wrong effect from find_first_of_1");
        }
        else if (n >= 2 && n < 10)
        {
            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1,
                                            first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_2");

            // No matches
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_3");
        }
        else if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto pos1 = n / 5;
            auto pos2 = 3 * n / 5;
            auto num = 3;

            ::std::iota(host_keys.get() + pos2, host_keys.get() + pos2 + num, T1(7));
            host_keys.update_data();

            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first1 + pos2, "Wrong effect from find_first_of_4");

            // Add second match
            ::std::iota(host_keys.get() + pos1, host_keys.get() + pos1 + num, T1(6));
            host_keys.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first1 + pos1, "Wrong effect from find_first_of_5");
        }
    }
};

DEFINE_TEST(test_search)
{
    DEFINE_TEST_CONSTRUCTOR(test_search)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(5));
        ::std::iota(host_vals.get(), host_vals.get() + n, T1(0));
        update_data(host_keys, host_vals);

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from search_00");
            res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from search_01");
        }
        auto res1 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1 == last1, "wrong effect from search_1");
        if (n > 10)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::search(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 10, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res2 - first1 == 5, "wrong effect from search_2");
        }
        // subsequence consists of one element (last one)
        auto res3 = ::std::search(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, last1 - 1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(last1 - res3 == 1, "wrong effect from search_3");

        // first sequence contains 2 almost similar parts
        if (n > 5)
        {
            ::std::iota(host_keys.get() + n / 2, host_keys.get() + n, T1(5));
            host_keys.update_data();

            auto res4 = ::std::search(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2 + 5, first2 + 6);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res4 == first1, "wrong effect from search_4");
        }
    }
};

DEFINE_TEST(test_search_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_search_n)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        ::std::iota(host_keys.get(), host_keys.get() + n, T(5));

        // Search for sequence at the end
        {
            auto start = (n > 3) ? (n / 3 * 2) : (n - 1);

            ::std::fill(host_keys.get() + start, host_keys.get() + n, T(11));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, n - start, T(11));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_1");
        }
        // Search for sequence in the middle
        {
            auto start = (n > 3) ? (n / 3) : (n - 1);
            auto end = (n > 3) ? (n / 3 * 2) : n;

            ::std::fill(host_keys.get() + start, host_keys.get() + end, T(22));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, end - start, T(22));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_20");

            // Search for sequence of lesser size
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                ::std::max(end - start - 1, (size_t)1), T(22));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_21");
        }
        // Search for sequence at the beginning
        {
            auto end = n / 3;

            ::std::fill(host_keys.get(), host_keys.get() + end, T(33));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last, end, T(33));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_3");
        }
        // Search for sequence that covers the whole range
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T(44));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, n, T(44));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_4");
        }
        // Search for sequence which is not there
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 5>>(exec), first, last, 2, T(55));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_50");

            // Sequence is there but of lesser size(see search_n_3)
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 6>>(exec), first, last, (n / 3 + 1), T(33));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_51");
        }

        // empty sequence case
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 7>>(exec), first, first, 1, T(5 + n - 1));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_6");
        }
        // 2 distinct sequences, must find the first one
        if (n > 10)
        {
            auto start1 = n / 6;
            auto end1 = n / 3;

            auto start2 = (2 * n) / 3;
            auto end2 = (5 * n) / 6;

            ::std::fill(host_keys.get() + start1, host_keys.get() + end1, T(66));
            ::std::fill(host_keys.get() + start2, host_keys.get() + end2, T(66));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 8>>(exec), first, last,
                                     ::std::min(end1 - start1, end2 - start2), T(66));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start1, "wrong effect from search_7");
        }

        if (n == 10)
        {
            auto seq_len = 3;

            // Should fail when searching for sequence which is placed before our first iterator.
            ::std::fill(host_keys.get(), host_keys.get() + seq_len, T(77));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 9>>(exec), first + 1, last, seq_len, T(77));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_8");
        }
    }
};

DEFINE_TEST(test_mismatch)
{
    DEFINE_TEST_CONSTRUCTOR(test_mismatch)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(5));
        ::std::iota(host_vals.get(), host_vals.get() + n, T1(0));
        update_data(host_keys, host_vals);

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_00");
            res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_01");
        }
        auto res1 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1.first == first1 && res1.second == first2, "wrong effect from mismatch_1");
        if (n > 5)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 5, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res2.first == last1 - 5 && res2.second == last2, "wrong effect from mismatch_2");
        }
    }
};

DEFINE_TEST(test_transform_inclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_inclusive_scan)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto res1 = ::std::transform_inclusive_scan(
            make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, ::std::plus<T1>(),
            [](T1 x) { return x * 2; }, value);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from transform_inclusive_scan_1");

        retrieve_data(host_keys, host_vals);

        T1 ii = value;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2 * host_keys.get()[i];
            if (host_vals.get()[i] != ii)
            {
                ::std::cout << "Error in scan_1: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i]
                            << ::std::endl;
            }
            EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_inclusive_scan_1");
        }

        // without initial value
        auto res2 = ::std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1,
                                                    first2, ::std::plus<T1>(), [](T1 x) { return x * 2; });
        EXPECT_TRUE(res2 == last2, "wrong result from transform_inclusive_scan_2");

        retrieve_data(host_keys, host_vals);

        ii = 0;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2 * host_keys.get()[i];
            if (host_vals.get()[i] != ii)
            {
                ::std::cout << "Error in scan_2: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i]
                            << ::std::endl;
            }
            EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_inclusive_scan_2");
        }
    }
};

DEFINE_TEST(test_transform_exclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_exclusive_scan)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto res1 =
            ::std::transform_exclusive_scan(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2,
                                          T1{}, ::std::plus<T1>(), [](T1 x) { return x * 2; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from transform_exclusive_scan");

        auto ii = T1(0);

        retrieve_data(host_keys, host_vals);

        for (size_t i = 0; i < last2 - first2; ++i)
        {
            if (host_vals.get()[i] != ii)
                ::std::cout << "Error: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i] << ::std::endl;

            //EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_exclusive_scan");
            ii += 2 * host_keys.get()[i];
        }
    }
};

DEFINE_TEST(test_copy_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_if)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto res1 = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                                   [](T1 x) { return x > -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from copy_if_1");

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < res1 - first2; ++i)
        {
            auto exp = i + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_1");
        }

        auto res2 = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                 [](T1 x) { return x % 2 == 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res2 == first2 + (last2 - first2) / 2, "wrong result from copy_if_2");

        host_vals.retrieve_data();
        host_first2 = host_vals.get();
        for (int i = 0; i < res2 - first2; ++i)
        {
            auto exp = 2 * i + 1 + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_2: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_2");
        }
    }
};

DEFINE_TEST(test_remove)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;
        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = ::std::remove(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, T1(222 + pos));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last - 1, "wrong result from remove");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove");
        }
    }
};

DEFINE_TEST(test_remove_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove_if)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = ::std::remove_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                   [=](T1 x) { return x == T1(222 + pos); });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last - 1, "wrong result from remove_if");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove_if");
        }
    }
};

DEFINE_TEST(test_unique_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // init
        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator1ValueType{ -1 });
        update_data(host_keys, host_vals);

        // invoke
        auto f = [](Iterator1ValueType a, Iterator1ValueType b) { return a == b; };
        auto result_first = first2;
        auto result_last =
            ::std::unique_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, result_first, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto result_size = result_last - result_first;

        std::int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
#    if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "buffer size: got " << result_last - result_first << ", expected " << expected_size
                      << ::std::endl;
#    endif

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first2 + i) != i + 1)
            {
                is_correct = false;
                ::std::cout << "got: " << *(host_first2 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique_copy");
        }
    }
};

DEFINE_TEST(test_unique)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&index](IteratorValueType& value) { value = (index++ + 4) / 4; });
        host_keys.update_data();

        // invoke
        auto f = [](IteratorValueType a, IteratorValueType b) { return a == b; };
        auto result_last = ::std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto result_size = result_last - first;

        std::int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
#    if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "buffer size: got " << result_last - first << ", expected " << expected_size << ::std::endl;
#    endif

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first1 + i) != i + 1)
            {
                is_correct = false;
#    if _ONEDPL_DEBUG_SYCL
                ::std::cout << "got: " << *(host_first1 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
#    endif
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique");
        }
    }
};

DEFINE_TEST(test_partition_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;
        using Iterator3ValueType = typename ::std::iterator_traits<Iterator3>::value_type;
        auto f = [](Iterator1ValueType value) { return (value % 3 == 0) && (value % 2 == 0); };

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, Iterator1ValueType{0});
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator2ValueType{-1});
        ::std::fill(host_res.get(),   host_res.get() + n, Iterator3ValueType{-2});
        update_data(host_keys, host_vals, host_res);

        // invoke
        auto res =
            ::std::partition_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first3, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        retrieve_data(host_keys, host_vals, host_res);

        // init for expected
        ::std::vector<Iterator2ValueType> exp_true(n, -1);
        ::std::vector<Iterator3ValueType> exp_false(n, -2);
        auto exp_true_first = exp_true.begin();
        auto exp_false_first = exp_false.begin();

        // invoke for expected
        auto exp = ::std::partition_copy(host_keys.get(), host_keys.get() + n, exp_true_first, exp_false_first, f);

        // check
        bool is_correct = (exp.first - exp_true_first) == (res.first - first2) &&
                          (exp.second - exp_false_first) == (res.second - first3);
#    if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "N =" << n << ::std::endl
                      << "buffer size: got {" << res.first - first2 << "," << res.second - first3 << "}, expected {"
                      << exp.first - exp_true_first << "," << exp.second - exp_false_first << "}" << ::std::endl;
#    endif

        for (int i = 0; i < ::std::min(exp.first - exp_true_first, res.first - first2) && is_correct; ++i)
        {
            if (*(exp_true_first + i) != *(host_vals.get() + i))
            {
                is_correct = false;
#    if _ONEDPL_DEBUG_SYCL
                ::std::cout << "TRUE> got: " << *(host_vals.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_true_first + i) << "[" << i << "]" << ::std::endl;
#    endif
            }
        }

        for (int i = 0; i < ::std::min(exp.second - exp_false_first, res.second - first3) && is_correct; ++i)
        {
            if (*(exp_false_first + i) != *(host_res.get() + i))
            {
                is_correct = false;
#    if _ONEDPL_DEBUG_SYCL
                ::std::cout << "FALSE> got: " << *(host_res.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_false_first + i) << "[" << i << "]" << ::std::endl;
#    endif
            }
        }

        EXPECT_TRUE(is_correct, "wrong effect from partition_copy");
    }
};

DEFINE_TEST(test_partition)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{ 0 });
        host_keys.update_data();

        // invoke partition
        auto unary_op = [](IteratorValueType value) { return (value % 3 == 0) && (value % 2 == 0); };
        auto res = ::std::partition(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, unary_op);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        // check
        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op),
                    "wrong effect from partition");
        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{0});
        host_keys.update_data();

        // invoke stable_partition
        res = ::std::stable_partition(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, unary_op);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op) &&
                        ::std::is_sorted(host_keys.get(), host_keys.get() + (res - first)) &&
                        ::std::is_sorted(host_keys.get() + (res - first), host_keys.get() + n),
                    "wrong effect from stable_partition");
    }
};

DEFINE_TEST(test_is_heap_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap_until)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        // first element is always a heap
        EXPECT_TRUE(actual == first + 1, "wrong result of is_heap_until_1");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == (first + n / 2), "wrong result of is_heap_until_2");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == last, "wrong result of is_heap_until_3");
    }
};

DEFINE_TEST(test_is_heap)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        // True only when n == 1
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == (n == 1), "wrong result of is_heap_11");

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == true, "wrong result of is_heap_12");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == false, "wrong result of is_heap_21");

        auto end = first + n / 2;
        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, end);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == true, "wrong result of is_heap_22");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == true, "wrong result of is_heap_3");
    }
};

DEFINE_TEST(test_inplace_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_inplace_merge)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;
        auto value = T(0);

        ::std::iota(host_keys.get(), host_keys.get() + n, value);

        ::std::vector<T> exp(n);
        ::std::iota(exp.begin(), exp.end(), value);

        auto middle = ::std::stable_partition(host_keys.get(), host_keys.get() + n, [](const T& x) { return x % 2; });
        host_keys.update_data();

        ::std::inplace_merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + (middle - host_keys.get()), last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_keys.retrieve_data();
        for (size_t i = 0; i < n; ++i)
        {
            if (host_keys.get()[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_keys.get()[i] << ::std::endl;
            }
            EXPECT_TRUE(host_keys.get()[i] == exp[i], "wrong effect from inplace_merge");
        }
    }
};

DEFINE_TEST(test_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_merge)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        typedef typename ::std::iterator_traits<Iterator3>::value_type T3;

        auto value = T1(0);
        auto x = n > 1 ? n / 2 : n;
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        ::std::iota(host_vals.get(), host_vals.get() + n, T2(value));
        update_data(host_keys, host_vals);

        ::std::vector<T3> exp(2 * n);
        auto exp1 = ::std::merge(host_keys.get(), host_keys.get() + n, host_vals.get(), host_vals.get() + x, exp.begin());
        auto res1 = ::std::merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2 + x, first3);
        TestDataTransfer<UDTKind::eRes, Size> host_res(*this, res1 - first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        // Special case, because we have more results then source data
        host_res.retrieve_data();
        auto host_first3 = host_res.get();
#    if _ONEDPL_DEBUG_SYCL
        for (size_t i = 0; i < res1 - first3; ++i)
        {
            if (host_first3[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_first3[i] << ::std::endl;
            }
        }
#    endif
        EXPECT_TRUE(res1 - first3 == exp1 - exp.begin(), "wrong result from merge_1");
        EXPECT_TRUE(::std::is_sorted(host_first3, host_first3 + (res1 - first3)), "wrong effect from merge_1");
    }
};
#endif

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_any_all_none_of");
    test1buffer<alloc_type, test_any_all_none_of<ValueType>>();
    PRINT_DEBUG("test_inplace_merge");
    test1buffer<alloc_type, test_inplace_merge<ValueType>>();
    PRINT_DEBUG("test_partition");
    test1buffer<alloc_type, test_partition<ValueType>>();
    PRINT_DEBUG("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();
    PRINT_DEBUG("test_find_if");
    test1buffer<alloc_type, test_find_if<ValueType>>();
    PRINT_DEBUG("test_search_n");
    test1buffer<alloc_type, test_search_n<ValueType>>();
    PRINT_DEBUG("test_remove");
    test1buffer<alloc_type, test_remove<ValueType>>();
    PRINT_DEBUG("test_remove_if");
    test1buffer<alloc_type, test_remove_if<ValueType>>();
    PRINT_DEBUG("test_unique");
    test1buffer<alloc_type, test_unique<ValueType>>();
    PRINT_DEBUG("test_is_heap_until");
    test1buffer<alloc_type, test_is_heap_until<ValueType>>();
    print_debug("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_equal");
    test2buffers<alloc_type, test_equal<ValueType>>();
    PRINT_DEBUG("test_mismatch");
    test2buffers<alloc_type, test_mismatch<ValueType>>();
    PRINT_DEBUG("test_search");
    test2buffers<alloc_type, test_search<ValueType>>();
    PRINT_DEBUG("test_transform_inclusive_scan");
    test2buffers<alloc_type, test_transform_inclusive_scan<ValueType>>();
    PRINT_DEBUG("test_transform_exclusive_scan");
    test2buffers<alloc_type, test_transform_exclusive_scan<ValueType>>();
    PRINT_DEBUG("test_copy_if");
    test2buffers<alloc_type, test_copy_if<ValueType>>();
    PRINT_DEBUG("test_unique_copy");
    test2buffers<alloc_type, test_unique_copy<ValueType>>();
    PRINT_DEBUG("test_find_first_of");
    test2buffers<alloc_type, test_find_first_of<ValueType>>();

    //test3buffers
    PRINT_DEBUG("test_partition_copy");
    test3buffers<alloc_type, test_partition_copy<ValueType>>();
    PRINT_DEBUG("test_merge");
    test3buffers<alloc_type, test_merge<ValueType>>(2);
}
#endif

std::int32_t
main()
{
    try
    {
#if TEST_DPCPP_BACKEND_PRESENT
        //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
        //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

        // Run tests for USM shared memory
        test_usm_and_buffer<sycl::usm::alloc::shared>();
        // Run tests for USM device memory
        test_usm_and_buffer<sycl::usm::alloc::device>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
