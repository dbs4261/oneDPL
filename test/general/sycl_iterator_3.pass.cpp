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

DEFINE_TEST(test_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_sort)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(333);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        {
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
#           if _ONEDPL_DEBUG_SYCL
            for (int i = 0; i < n; ++i)
            {
                if (host_first1[i] != value + i)
                {
                    ::std::cout << "Error_1. i = " << i << ", expected = " << value + i << ", got = " << host_first1[i]
                                << ::std::endl;
                }
            }
#           endif
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from sort_1");
        }

        ::std::sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
#       if _ONEDPL_DEBUG_SYCL
        for (int i = 0; i < n; ++i)
        {
            if (host_first1[i] != value + n - 1 - i)
            {
                ::std::cout << "Error_2. i = " << i << ", expected = " << value + n - 1 - i
                          << ", got = " << host_first1[i] << ::std::endl;
            }
        }
#       endif
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()), "wrong effect from sort_2");
    }
};

DEFINE_TEST(test_stable_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_stable_sort)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(333);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        {
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();

#           if _ONEDPL_DEBUG_SYCL
            for (int i = 0; i < n; ++i)
            {
                if (host_first1[i] != value + i)
                {
                    ::std::cout << "Error_1. i = " << i << ", expected = " << value + i << ", got = " << host_first1[i]
                              << ::std::endl;
                }
            }
#           endif
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from stable_sort_1");
        }

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
#       if _ONEDPL_DEBUG_SYCL
        for (int i = 0; i < n; ++i)
        {
            if (host_first1[i] != value + n - 1 - i)
            {
                ::std::cout << "Error_2. i = " << i << ", expected = " << value + n - 1 - i
                            << ", got = " << host_first1[i] << ::std::endl;
            }
        }
#       endif
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()),
                    "wrong effect from stable_sort_3");
    }
};

DEFINE_TEST(test_partial_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_partial_sort)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* first1 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        if (n <= 1)
            return;

        auto value = T1(333);
        auto init = value;
        ::std::generate(host_keys.get(), host_keys.get() + n, [&init]() { return init--; });
        host_keys.update_data();

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end1 = first1 + end_idx;
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, end1, last1);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif

            // Make sure that elements up to end are sorted and remaining elements are bigger
            // than the last sorted one.
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + end_idx), "wrong effect from partial_sort_1");

            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n,
                                   [&](T1 val) { return val >= *(host_first1 + end_idx - 1); });
            EXPECT_TRUE(res, "wrong effect from partial_sort_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, last1);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from partial_sort_2");
        }
    }
};

DEFINE_TEST(test_partial_sort_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_partial_sort_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);

        if (n <= 1)
            return;

        auto init = value;
        ::std::generate(host_keys.get(), host_keys.get() + n, [&init]() { return init--; });
        host_keys.update_data();

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end2 = first2 + end_idx;

            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, end2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            EXPECT_TRUE(last_sorted == end2, "wrong effect from partial_sort_copy_1");
            // Make sure that elements up to end2 are sorted
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + end_idx), "wrong effect from partial_sort_copy_1");

            // Now ensure that the original sequence wasn't changed by partial_sort_copy
            auto init = value;
            auto res = ::std::all_of(host_first1, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            EXPECT_TRUE(last_sorted == last2, "wrong effect from partial_sort_copy_2");
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + n), "wrong effect from partial_sort_copy_2");

            // Now ensure that partial_sort_copy hasn't change the unsorted part of original sequence
            auto init = value - end_idx;
            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_2");
        }
    }
};

DEFINE_TEST(test_find_end)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_end)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;

        // Reset after previous run
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        }

        if (n <= 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            host_vals.update_data();

            // Empty subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_end_1");

            return;
        }

        if (n > 2 && n < 10)
        {
            // re-write the sequence after previous run
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            update_data(host_keys, host_vals);

            // No subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2 + n / 2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_end_2");

            // Whole sequence is matched
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(10));
            host_keys.update_data();

            res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first1, "Wrong effect from find_end_3");

            return;
        }

        if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n / 5, T2(20));
            host_vals.update_data();

            // Match at the beginning
            {
                auto start = host_keys.get();
                auto end = host_keys.get() + n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                           first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1, "Wrong effect from find_end_4");
            }

            // 2 matches: at the beginning and in the middle, should return the latter
            {
                auto start = host_keys.get() + 2 * n / 5;
                auto end = host_keys.get() + 3 * n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();


                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                           first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + 2 * n / 5, "Wrong effect from find_end_5");
            }

            // 3 matches: at the beginning, in the middle and at the end, should return the latter
            {
                auto start = host_keys.get() + 4 * n / 5;
                auto end = host_keys.get() + n;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                           first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + 4 * n / 5, "Wrong effect from find_end_6");
            }
        }
    }
};

// TODO: move unique cases to test_lexicographical_compare
DEFINE_TEST(test_lexicographical_compare)
{
    DEFINE_TEST_CONSTRUCTOR(test_lexicographical_compare)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // INIT
        {
            ValueType fill_value1{0};
            ::std::for_each(host_keys.get(), host_keys.get() + n,
                            [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
            ValueType fill_value2{0};
            ::std::for_each(host_vals.get(), host_vals.get() + n,
                            [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
            update_data(host_keys, host_vals);
        }

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

        // CHECK 1.1: S1 == S2 && len(S1) == len(S2)
        bool is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1,
                                                          last1, first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.1: S1 == S2 && len(S1) == len(S2)");

        // CHECK 1.2: S1 == S2 && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 1.2: S1 == S2 && len(S1) < len(S2)");

        // CHECK 1.3: S1 == S2 && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.3: S1 == S2 && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_vals.get() + n - 2) = 222;
            host_vals.update_data();
        }

        // CHECK 2.1: S1 < S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        bool is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.1: S1 < S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 2.2: S1 < S2 (PRE-LAST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.2: S1 < S2 (PRE-LAST) && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_keys.get() + n - 2) = 333;
            host_keys.update_data();
        }

        // CHECK 3.1: S1 > S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0,
                    "wrong effect from lex_compare Test 3.1: S1 > S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 3.2: S1 > S2 (PRE-LAST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 3.2: S1 > S2 (PRE-LAST) && len(S1) < len(S2)");
        {
            *host_vals.get() = 444;
            host_vals.update_data();
        }

        // CHECK 4.1: S1 < S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 4.1: S1 < S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 4.2: S1 < S2 (FIRST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 8>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 4.2: S1 < S2 (FIRST) && len(S1) > len(S2)");
        {
            *host_keys.get() = 555;
            host_keys.update_data();
        }

        // CHECK 5.1: S1 > S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 9>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 5.1: S1 > S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 5.2: S1 > S2 (FIRST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 10>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 5.2: S1 > S2 (FIRST) && len(S1) < len(S2)");
    }
};

DEFINE_TEST(test_swap_ranges)
{
    DEFINE_TEST_CONSTRUCTOR(test_swap_ranges)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using value_type = typename ::std::iterator_traits<Iterator1>::value_type;
        using reference = typename ::std::iterator_traits<Iterator1>::reference;

        ::std::iota(host_keys.get(), host_keys.get() + n, value_type(0));
        ::std::iota(host_vals.get(), host_vals.get() + n, value_type(n));
        update_data(host_keys, host_vals);

        Iterator2 actual_return = ::std::swap_ranges(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        bool check_return = (actual_return == last2);
        EXPECT_TRUE(check_return, "wrong result of swap_ranges");
        if (check_return)
        {
            ::std::size_t i = 0;

            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();
            bool check =
                ::std::all_of(host_first2, host_first2 + n, [&i](reference a) { return a == value_type(i++); }) &&
                ::std::all_of(host_first1, host_first1 + n, [&i](reference a) { return a == value_type(i++); });

            EXPECT_TRUE(check, "wrong effect of swap_ranges");
        }
    }
};

DEFINE_TEST(test_nth_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_nth_element)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T1 = typename ::std::iterator_traits<Iterator1>::value_type;
        using T2 = typename ::std::iterator_traits<Iterator2>::value_type;

        // init
        auto value1 = T1(0);
        auto value2 = T2(0);
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&value1](T1& val) { val = (value1++ % 10) + 1; });
        ::std::for_each(host_vals.get(), host_vals.get() + n, [&value2](T2& val) { val = (value2++ % 10) + 1; });
        update_data(host_keys, host_vals);

        auto middle1 = first1 + n / 2;

        // invoke
        auto comp = ::std::less<T1>{};
        ::std::nth_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, middle1, last1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        ::std::nth_element(host_first2, host_first2 + n / 2, host_first2 + n, comp);

        // check
        auto median = *(host_first1 + n / 2);
        bool is_correct = median == *(host_first2 + n / 2);
        if (!is_correct)
        {
            ::std::cout << "wrong nth element value got: " << median << ", expected: " << *(host_first2 + n / 2)
                      << ::std::endl;
        }
        is_correct =
            ::std::find_first_of(host_first1, host_first1 + n / 2, host_first1 + n / 2, host_first1 + n,
                               [comp](T1& x, T2& y) { return comp(y, x); }) ==
                     host_first1 + n / 2;
        EXPECT_TRUE(is_correct, "wrong effect from nth_element");
    }
};

DEFINE_TEST(test_reverse)
{
    DEFINE_TEST_CONSTRUCTOR(test_reverse)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::reverse(local_copy.begin(), local_copy.end());

        ::std::reverse(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from reverse");
    }
};

DEFINE_TEST(test_reverse_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_reverse_copy)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 /* result_last */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::reverse(local_copy.begin(), local_copy.end());

        ::std::reverse_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, result_first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_first2[i], "wrong effect from reverse_copy");
    }
};

DEFINE_TEST(test_rotate)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from rotate");
    }
};

DEFINE_TEST(test_rotate_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate_copy)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 /* result_last */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last, result_first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_vals.get()[i], "wrong effect from rotate_copy");
    }
};

int a[] = {0, 0, 1, 1, 2, 6, 6, 9, 9};
int b[] = {0, 1, 1, 6, 6, 9};
int c[] = {0, 1, 6, 6, 6, 9, 9};
int d[] = {7, 7, 7, 8};
int e[] = {11, 11, 12, 16, 19};
constexpr size_t count_a = sizeof(a) / sizeof(a[0]);
constexpr size_t count_b = sizeof(b) / sizeof(b[0]);
constexpr size_t count_c = sizeof(c) / sizeof(c[0]);
constexpr size_t count_d = sizeof(d) / sizeof(d[0]);
constexpr size_t count_abcd = count_a + count_b + count_c + count_d;


DEFINE_TEST(test_includes)
{
    DEFINE_TEST_CONSTRUCTOR(test_includes)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        if (n < count_abcd)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        //first test case
        last1 = first1 + count_a;
        last2 = first2 + count_b;

        ::std::copy(a, a + count_a, host_keys.get());
        ::std::copy(b, b + count_b, host_vals.get());
        host_keys.update_data(count_a);
        host_vals.update_data(count_b);

        auto result = ::std::includes(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result, "wrong effect from includes a, b");

        host_vals.retrieve_data();
        ::std::copy(c, c + count_c, host_vals.get());
        host_vals.update_data(count_c);

        result = ::std::includes(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        EXPECT_TRUE(!result, "wrong effect from includes a, c");
    }
};

DEFINE_TEST(test_set_intersection)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_intersection)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        if (n < count_abcd)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        //first test case
        last1 = first1 + count_a;
        last2 = first2 + count_b;
        ::std::copy(a, a + count_a, host_keys.get());
        ::std::copy(b, b + count_b, host_vals.get());
        host_keys.update_data(count_a);
        host_vals.update_data(count_b);

        last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2,
                                      first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_res.retrieve_data();
        auto nres = last3 - first3;

        EXPECT_TRUE(nres == 6, "wrong size of intersection of a, b");

        auto result = ::std::includes(host_keys.get(), host_keys.get() + count_a, host_res.get(), host_res.get() + nres) &&
                      ::std::includes(host_vals.get(), host_vals.get() + count_b, host_res.get(), host_res.get() + nres);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result, "wrong effect from set_intersection a, b");

        { //second test case

            last2 = first2 + count_d;
            ::std::copy(a, a + count_a, host_keys.get());
            ::std::copy(d, d + count_d, host_vals.get());
            host_keys.update_data(count_a);
            host_vals.update_data(count_b);

            last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                          last2, first3);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            auto nres = last3 - first3;
            EXPECT_TRUE(nres == 0, "wrong size of intersection of a, d");
        }
    }
};

DEFINE_TEST(test_set_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_difference)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        if (n < count_abcd)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        last1 = first1 + count_a;
        last2 = first2 + count_b;

        ::std::copy(a, a + count_a, host_keys.get());
        ::std::copy(b, b + count_b, host_vals.get());
        host_keys.update_data(count_a);
        host_vals.update_data(count_b);

        last3 = ::std::set_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        int res_expect[count_a];
        host_res.retrieve_data();
        auto nres_expect = ::std::set_difference(host_keys.get(), host_keys.get() + count_a, host_vals.get(), host_vals.get() + count_b, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_difference a, b");
    }
};

DEFINE_TEST(test_set_union)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_union)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        if (n < count_abcd)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        last1 = first1 + count_a;
        last2 = first2 + count_b;

        ::std::copy(a, a + count_a, host_keys.get());
        ::std::copy(b, b + count_b, host_vals.get());
        host_keys.update_data(count_a);
        host_vals.update_data(count_b);

        last3 = ::std::set_union(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        int res_expect[count_a + count_b];
        host_res.retrieve_data();
        auto nres_expect =
            ::std::set_union(host_keys.get(), host_keys.get() + count_a, host_vals.get(), host_vals.get() + count_b, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_union a, b");
    }
};

DEFINE_TEST(test_set_symmetric_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_symmetric_difference)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        if (n < count_abcd)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes, Size>  host_res (*this, n);

        last1 = first1 + count_a;
        last2 = first2 + count_b;

        ::std::copy(a, a + count_a, host_keys.get());
        ::std::copy(b, b + count_b, host_vals.get());
        host_keys.update_data(count_a);
        host_vals.update_data(count_b);

        last3 = ::std::set_symmetric_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                                                first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        int res_expect[count_a + count_b];
        retrieve_data(host_keys, host_vals, host_res);
        auto nres_expect = ::std::set_symmetric_difference(host_keys.get(), host_keys.get() + count_a, host_vals.get(),
                                                           host_vals.get() + count_b, res_expect) -
                           res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_symmetric_difference a, b");
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
    PRINT_DEBUG("test_sort");
    test1buffer<alloc_type, test_sort<ValueType>>();
    PRINT_DEBUG("test_reverse");
    test1buffer<alloc_type, test_reverse<ValueType>>();
    PRINT_DEBUG("test_rotate");
    test1buffer<alloc_type, test_rotate<ValueType>>();
    PRINT_DEBUG("test_stable_sort");
    test1buffer<alloc_type, test_stable_sort<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_nth_element");
    test2buffers<alloc_type, test_nth_element<ValueType>>();
    PRINT_DEBUG("test_swap_ranges");
    test2buffers<alloc_type, test_swap_ranges<ValueType>>();
    PRINT_DEBUG("test_reverse_copy");
    test2buffers<alloc_type, test_reverse_copy<ValueType>>();
    PRINT_DEBUG("test rotate_copy");
    test2buffers<alloc_type, test_rotate_copy<ValueType>>();
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<alloc_type, test_lexicographical_compare<ValueType>>();
    PRINT_DEBUG("test_partial_sort");
    test2buffers<alloc_type, test_partial_sort<ValueType>>();
    PRINT_DEBUG("test_partial_sort_copy");
    test2buffers<alloc_type, test_partial_sort_copy<ValueType>>();
    PRINT_DEBUG("test_find_end");
    test2buffers<alloc_type, test_find_end<ValueType>>();
    PRINT_DEBUG("test_includes");
    test2buffers<alloc_type, test_includes<ValueType>>();

    //test3buffers
    PRINT_DEBUG("test_set_symmetric_difference");
    test3buffers<alloc_type, test_set_symmetric_difference<ValueType>>();
    PRINT_DEBUG("test_set_union");
    test3buffers<alloc_type, test_set_union<ValueType>>();
    PRINT_DEBUG("test_set_difference");
    test3buffers<alloc_type, test_set_difference<ValueType>>();
    PRINT_DEBUG("test_set_intersection");
    test3buffers<alloc_type, test_set_intersection<ValueType>>();
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
