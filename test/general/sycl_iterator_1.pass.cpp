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

struct Plus
{
    template <typename T, typename U>
    T
    operator()(const T x, const U y) const
    {
        return x + y;
    }
};

using namespace oneapi::dpl::execution;

DEFINE_TEST(test_replace)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, value, T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace");
    }
};

DEFINE_TEST(test_replace_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_if)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                          oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace_if");
    }
};

DEFINE_TEST(test_replace_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, value, T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value + 1),
                    "wrong effect from replace_copy");
    }
};

DEFINE_TEST(test_replace_copy_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_copy_if)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                             oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value + 1),
                    "wrong effect from replace_copy_if");
    }
};

DEFINE_TEST(test_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{0});
        update_data(host_keys, host_vals);

        ::std::copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from copy");
    }
};

DEFINE_TEST(test_copy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_n)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ 0 });
        update_data(host_keys, host_vals);

        ::std::copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from copy_n");
    }
};

DEFINE_TEST(test_move)
{
    DEFINE_TEST_CONSTRUCTOR(test_move)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ 0 });
        update_data(host_keys, host_vals);

        ::std::move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from move");
    }
};

DEFINE_TEST(test_adjacent_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_difference)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        Iterator1ValueType fill_value{1};
        Iterator2ValueType blank_value{0};

        auto __f = [](Iterator1ValueType& a, Iterator1ValueType& b) -> Iterator2ValueType { return a + b; };

        // init
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](Iterator1ValueType& val) { val = (fill_value++ % 10) + 1; });
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        update_data(host_keys, host_vals);

        // test with custom functor
        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, __f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        {
            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            bool is_correct = *host_first1 == *host_first2; // for the first element
            for (int i = 1; i < n; ++i)                     // for subsequent elements
                is_correct = is_correct && *(host_first2 + i) == __f(*(host_first1 + i), *(host_first1 + i - 1));

            EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #1");
        }

        // test with default functor
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        host_vals.update_data();

        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        bool is_correct = *host_first1 == *host_first2; // for the first element
        for (int i = 1; i < n; ++i)                     // for subsequent elements
            is_correct = is_correct && *(host_first2 + i) == (*(host_first1 + i) - *(host_first1 + i - 1));

        EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #2");
    }
};

DEFINE_TEST(test_reduce)
{
    DEFINE_TEST_CONSTRUCTOR(test_reduce)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value);
        host_keys.update_data();

        // without initial value
        auto result1 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result1 == value * (n / 2 - n / 3), "wrong effect from reduce (1)");

        // with initial value
        auto init = T1(42);
        auto result2 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), init);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result2 == init + value * (n / 2 - n / 3), "wrong effect from reduce (2)");
    }
};

DEFINE_TEST(test_transform_reduce_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_unary)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(1);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto result = ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, T1(42),
                                            Plus(), ::std::negate<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == 42 - n, "wrong effect from transform_reduce (unary + binary)");
    }
};

DEFINE_TEST(test_transform_reduce_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_binary)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* firs2 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(1);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto result =
            ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first1, T1(42));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == n + 42, "wrong effect from transform_reduce (2 binary)");
    }
};

DEFINE_TEST(test_min_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_min_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{ static_cast<IteratorValueType>(::std::distance(first, last)) };

        ::std::for_each(host_keys.get(), host_keys.get() + n,
            [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t min_dis = n;
        if (min_dis)
        {
            *(host_keys.get() + min_dis / 2) = IteratorValueType{/*min_val*/ 0 };
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd min*/ 0 };
        }
        host_keys.update_data();

        auto result_min = ::std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();

        auto expected_min = ::std::min_element(host_keys.get(), host_keys.get() + n);

        EXPECT_TRUE(result_min - first == expected_min - host_keys.get(),
                    "wrong effect from min_element");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << *(host_keys.get() + (result_min - first)) << "["
                    << result_min - first << "], "
                    << "expected: " << *(host_keys.get() + (expected_min - host_keys.get())) << "["
                    << expected_min - host_keys.get() << "]" << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_adjacent_find)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_find)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::equal_to<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check with no adjacent equal elements
        Iterator result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #1 no elements)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                    << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check with the last adjacent element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = *(host_keys.get() + n - 2);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 2 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #2 the last element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check with an adjacent element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + max_dis / 2) = *(host_keys.get() + max_dis / 2 - 1);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? it - 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #3 middle element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif
        // check with an adjacent element (no predicate)
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #4 middle element (no predicate))");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check with the first adjacent element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = *host_keys.get();
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? first : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #5 the first element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_max_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_max_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t max_dis = n;
        if (max_dis)
        {
            *(host_keys.get() + max_dis / 2) = IteratorValueType{/*max_val*/ 777};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd max*/ 777};
        }
        host_keys.update_data();

        auto expected_max_offset = ::std::max_element(host_keys.get(), host_keys.get() + n) - host_keys.get();

        auto result_max_offset = ::std::max_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last) - first;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_keys.retrieve_data();

        EXPECT_TRUE(result_max_offset == expected_max_offset, "wrong effect from max_element");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: "      << *(host_keys.get() + result_max_offset)   << "[" << result_max_offset   << "], "
                    << "expected: " << *(host_keys.get() + expected_max_offset) << "[" << expected_max_offset << "]"
                    << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_is_sorted_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted_until)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        Iterator result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from is_sorted_until (Test #1 sorted sequence)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #2 unsorted sequence - the last element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check unsorted: the middle element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + /*max_idx*/ max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected = it;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #3 unsorted sequence - the middle element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif
        // check unsorted: the middle element (no predicate)
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(
            result == expected,
            "wrong effect from is_sorted_until (Test #4 unsorted sequence - the middle element (no predicate))");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif

        // check unsorted: the first element
        if (n > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected = n > 1 ? first + 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #5 unsorted sequence - the first element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_minmax_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_minmax_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto fill_value = IteratorValueType{ 0 };

        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](IteratorValueType& it) { it = fill_value++ % 10 + 1; });
        ::std::size_t dis = n;
        if (dis > 1)
        {
            auto min_it = host_keys.get() + /*min_idx*/ dis / 2 - 1;
            *(min_it) = IteratorValueType{/*min_val*/ 0 };

            auto max_it = host_keys.get() + /*max_idx*/ dis / 2;
            *(max_it) = IteratorValueType{/*max_val*/ 777 };
        }
        host_keys.update_data();

        auto expected = ::std::minmax_element(host_keys.get(), host_keys.get() + n);
        auto expected_min = expected.first - host_keys.get();
        auto expected_max = expected.second - host_keys.get();
        ::std::pair<Size, Size> expected_offset = { expected_min, expected_max };

        auto result = ::std::minmax_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        auto result_min = result.first - first;
        auto result_max = result.second - first;

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_min == expected_min && result_max == expected_max, "wrong effect from minmax_element");
        if (!(result_min == expected_min && result_max == expected_max))
        {
            host_keys.retrieve_data();

            auto got_min = host_keys.get() + (result.first - first);
            auto got_max = host_keys.get() + (result.second - first);
            ::std::cout << "MIN got: " << got_min << "[" << result_min << "], "
                        << "expected: " << *(host_keys.get() + expected_offset.first) << "[" << expected_min << "]" << ::std::endl;
            ::std::cout << "MAX got: " << got_max << "[" << result_max << "], "
                        << "expected: " << *(host_keys.get() + expected_offset.second) << "[" << expected_max << "]" << ::std::endl;
        }
    }
};

DEFINE_TEST(test_is_sorted)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{ 0 };
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        bool expected_bool = true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_sorted (Test #1 sorted sequence)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#    endif

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #2 unsorted sequence - the last element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#    endif

        // check unsorted: the middle element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #3 unsorted sequence - the middle element)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#    endif
        // check unsorted: the middle element (no predicate)
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #4 unsorted sequence - the middle element (no predicate))");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#    endif

        // check unsorted: the first element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted Test #5 unsorted sequence - the first element");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_count)
{
    DEFINE_TEST_CONSTRUCTOR(test_count)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{0});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #1 arbitrary to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif

        // check when none should be counted
        expected = 0;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{12});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #2 none to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif

        // check when all should be counted
        ::std::fill(host_keys.get(), host_keys.get() + n, ValueType{7});
        host_keys.update_data();

        expected = n;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{7});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #3 all to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_count_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_count_if)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                            [](ValueType const& value) { return value % 10 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #1 arbitrary to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif

        // check when none should be counted
        expected = 0;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last,
                                 [](ValueType const& value) { return value > 10; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #2 none to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif

        // check when all should be counted
        expected = n;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                 [](ValueType const& value) { return value < 10; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #3 all to count)");
#    if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#    endif
    }
};

DEFINE_TEST(test_is_partitioned)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_partitioned)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        if (n < 2)
            return;

        auto less_than = [](const ValueType& value) -> bool { return value < 10; };
        auto is_odd = [](const ValueType& value) -> bool { return value % 2; };

        bool expected_bool_less_then = false;
        bool expected_bool_is_odd = false;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = ++fill_value; });
        expected_bool_less_then = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, less_than);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, less_than);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool_less_then, "wrong effect from is_partitioned (Test #1 less than)");

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, is_odd);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool_is_odd, "wrong effect from is_partitioned (Test #2 is odd)");

        // The code as below was added to prevent accessor destruction working with host memory
        ::std::partition(host_keys.get(), host_keys.get() + n, is_odd);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, is_odd);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool_is_odd, "wrong effect from is_partitioned (Test #3 is odd after partition)");
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
    PRINT_DEBUG("test_replace");
    test1buffer<alloc_type, test_replace<ValueType>>();
    PRINT_DEBUG("test_replace_if");
    test1buffer<alloc_type, test_replace_if<ValueType>>();
    PRINT_DEBUG("test_reduce");
    test1buffer<alloc_type, test_reduce<ValueType>>();
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<alloc_type, test_transform_reduce_unary<ValueType>>();
    PRINT_DEBUG("test_is_sorted");
    test1buffer<alloc_type, test_is_sorted<ValueType>>();
    PRINT_DEBUG("test_count");
    test1buffer<alloc_type, test_count<ValueType>>();
    PRINT_DEBUG("test_count_if");
    test1buffer<alloc_type, test_count_if<ValueType>>();
    PRINT_DEBUG("test_is_partitioned");
    test1buffer<alloc_type, test_is_partitioned<ValueType>>();
    PRINT_DEBUG("test_min_element");
    test1buffer<alloc_type, test_min_element<ValueType>>();
    PRINT_DEBUG("test_max_element");
    test1buffer<alloc_type, test_max_element<ValueType>>();
    PRINT_DEBUG("test_minmax_element");
    test1buffer<alloc_type, test_minmax_element<ValueType>>();
    PRINT_DEBUG("test_adjacent_find");
    test1buffer<alloc_type, test_adjacent_find<ValueType>>();
    PRINT_DEBUG("test_is_sorted_until");
    test1buffer<alloc_type, test_is_sorted_until<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_replace_copy");
    test2buffers<alloc_type, test_replace_copy<ValueType>>();
    PRINT_DEBUG("test_replace_copy_if");
    test2buffers<alloc_type, test_replace_copy_if<ValueType>>();
    PRINT_DEBUG("test_copy");
    test2buffers<alloc_type, test_copy<ValueType>>();
    PRINT_DEBUG("test_copy_n");
    test2buffers<alloc_type, test_copy_n<ValueType>>();
    PRINT_DEBUG("test_move");
    test2buffers<alloc_type, test_move<ValueType>>();
    PRINT_DEBUG("test_adjacent_difference");
    test2buffers<alloc_type, test_adjacent_difference<ValueType>>();
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<alloc_type, test_transform_reduce_binary<ValueType>>();
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
