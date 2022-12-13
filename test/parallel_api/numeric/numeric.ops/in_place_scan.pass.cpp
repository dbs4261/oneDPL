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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"
#include "support/scan_serial_impl.h"

// This macro may be used to analyze source data and test results in test_in_place_scan
// WARNING: in the case of using this macro debug output is very large.
#define DUMP_CHECK_RESULTS

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>

#    include <oneapi/tbb/global_control.h>
#endif
using namespace TestUtils;

template <typename T>
struct DefaultBinaryOp
{
    T operator()(T val1, T val2) const
    {
        return val1 + val2;
    }
};

DEFINE_TEST_1(test_inclusive_scan, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_inclusive_scan)

    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Iterator2, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Size n)
    {
        for (Size i = 0; i != n; ++i)
        {
            host_keys[i] = i;
            host_vals[i] = 0;
        }
    }

#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        std::cout << msg;
        for (Size i = 0; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    template <typename Iterator1, typename Iterator2, typename Size, typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(Iterator1 keys_first,
                      Iterator2 vals_first,
                      Size n,
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << n << ") : " << std::endl;
        display_param("keys:   ", keys_first, n);
        display_param("vals:   ", vals_first, n);
#endif // DUMP_CHECK_RESULTS

        if (n < 1)
            return;

        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;

        std::vector<ValT> expected(n);
        inclusive_scan_serial(keys_first, keys_first + n, expected.data(), op);

#ifdef DUMP_CHECK_RESULTS
        display_param("expected result: ", expected.data(), n);
#endif // DUMP_CHECK_RESULTS

        if (0 < EXPECT_EQ_N(expected.begin(), vals_first, n, "wrong effect from inclusive_scan_serial"))
        {
            // errors detected!

#ifdef DUMP_CHECK_RESULTS
            std::cout << "check_values(n = " << n << ") : " << std::endl;
            display_param("Source data:     ", keys_first, n);
            display_param("Actual results:  ", vals_first, n);
            display_param("Expected results:", expected, n);

#endif // DUMP_CHECK_RESULTS
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(host_keys.get(), host_vals.get(), n);
        update_data(host_keys, host_vals);

        // copy source host keys state into temp buf
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        const std::vector<ValT> source_host_keys_state(host_keys.get(), host_keys.get() + n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::inclusive_scan(new_policy, keys_first, keys_last, vals_first);
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_vals);
        check_values(host_keys.get(), host_vals.get(), n);

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), host_vals.get(), n);
        update_data(host_keys, host_vals);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::inclusive_scan(new_policy2, keys_first, keys_last, vals_first, DefaultBinaryOp<KeyT>());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_vals);
        check_values(host_keys.get(), host_vals.get(), n, DefaultBinaryOp<KeyT>());

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), host_vals.get(), n);
        update_data(host_keys, host_vals);

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::inclusive_scan(new_policy3, keys_first, keys_last, vals_first, BinaryOperation());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_vals);
        check_values(host_keys.get(), host_vals.get(), n, BinaryOperation());
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, n);
        auto res1 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, vals_first);
        check_values(keys_first, vals_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, n);
        auto res2 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, vals_first, DefaultBinaryOp<KeyT>());
        check_values(keys_first, vals_first, n);

        // call algorithm with addition operator
        initialize_data(keys_first, n);
        auto res3 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, vals_first, BinaryOperation());
        check_values(keys_first, vals_first, n, BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value, void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
    }
};

DEFINE_TEST_1(test_inclusive_scan_inplace, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_inclusive_scan_inplace)

    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Size>
    void
    initialize_data(Iterator1 host_keys, Size n)
    {
        for (Size i = 0; i != n; ++i)
            host_keys[i] = i;
    }

#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        std::cout << msg;
        for (Size i = 0; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    // Required to pass source_host_keys_state param by copy for safe state of this param on caller side
    template <typename ValT, typename Iterator1, typename Size,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(std::vector<ValT> source_host_keys_state, Iterator1 host_keys, Size n,
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
        const auto source_host_keys_state_copy = source_host_keys_state;

#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << source_host_keys_state.size() << ") : " << std::endl;
        display_param("keys:   ", host_keys, n);
#endif // DUMP_CHECK_RESULTS

        if (source_host_keys_state.empty())
            return;

        inclusive_scan_serial(source_host_keys_state.begin(), source_host_keys_state.end(), source_host_keys_state.begin(), op);

#ifdef DUMP_CHECK_RESULTS
        display_param("expected result: ", source_host_keys_state.data(), source_host_keys_state.size());
#endif // DUMP_CHECK_RESULTS

        if (0 < EXPECT_EQ_N(source_host_keys_state.cbegin(), host_keys, source_host_keys_state.size(), "wrong effect from inclusive_scan_serial"))
        {
            // errors detected!

#ifdef DUMP_CHECK_RESULTS
            std::cout << "check_values(n = " << source_host_keys_state.size() << ") : " << std::endl;
            display_param("Source data:     ", source_host_keys_state_copy, n);
            display_param("Actual results:  ", host_keys, n);
            display_param("Expected results:", source_host_keys_state, n);

#endif // DUMP_CHECK_RESULTS
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last,
               Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        // copy source host keys state into temp buf
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        const std::vector<ValT> source_host_keys_state(host_keys.get(), host_keys.get() + n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::inclusive_scan(new_policy, keys_first, keys_last, keys_first);
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n);

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::inclusive_scan(new_policy2, keys_first, keys_last, keys_first, DefaultBinaryOp<KeyT>());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n, DefaultBinaryOp<KeyT>());

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::inclusive_scan(new_policy3, keys_first, keys_last, keys_first, BinaryOperation());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n, BinaryOperation());
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, n);

        // copy source host keys state into temp buf
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        const std::vector<ValT> source_host_keys_state(keys_first, keys_first + n);

        auto res1 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, keys_first);
        check_values(source_host_keys_state, keys_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, n);
        auto res2 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, keys_first, DefaultBinaryOp<KeyT>());
        check_values(source_host_keys_state, keys_first, n, DefaultBinaryOp<KeyT>());

        // call algorithm with addition operator
        initialize_data(keys_first, n);
        auto res3 = oneapi::dpl::inclusive_scan(exec, keys_first, keys_last, keys_first, BinaryOperation());
        check_values(source_host_keys_state, keys_first, n, BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value, void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
    }
};

DEFINE_TEST_1(test_exclusive_scan_inplace, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_exclusive_scan_inplace)

    const int kInitValue = 2;

    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Size>
    void
    initialize_data(Iterator1 host_keys, Size n)
    {
        for (Size i = 0; i != n; ++i)
            host_keys[i] = i;
    }

#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        std::cout << msg << " : ";
        for (Size i = 0; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;

        //const Size kItemsPerLine = 20;

        //std::cout << msg << std::endl;
        //for (Size i = 0; i < n; ++i)
        //{
        //    if (i == 0 || (i > 0 && (i - 1) % kItemsPerLine == 0))
        //        std::cout << "\t";

        //    std::cout << it[i];
        //    if (i + 1 < n)
        //        std::cout << ", ";

        //    if (i % kItemsPerLine == 0)
        //        std::cout << std::endl;
        //}
        //std::cout << std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    // Required to pass source_host_keys_state param by copy for safe state of this param on caller side
    template <typename ValT, typename Iterator1, typename Size,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(std::vector<ValT> source_host_keys_state, Iterator1 host_keys, Size n,
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
        const auto source_host_keys_state_copy = source_host_keys_state;

#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << source_host_keys_state.size() << ") : " << std::endl;
        display_param("keys:   ", host_keys, n);
#endif // DUMP_CHECK_RESULTS

        if (source_host_keys_state.empty())
            return;

        exclusive_scan_serial(source_host_keys_state.begin(), source_host_keys_state.end(), source_host_keys_state.begin(), kInitValue, op);

#ifdef DUMP_CHECK_RESULTS
        display_param("expected result: ", source_host_keys_state.data(), source_host_keys_state.size());
#endif // DUMP_CHECK_RESULTS

        if (0 < EXPECT_EQ_N(source_host_keys_state.cbegin(), host_keys, source_host_keys_state.size(), "wrong effect from exclusive_scan_serial"))
        {
            // errors detected!

#ifdef DUMP_CHECK_RESULTS
            std::cout << "check_values(n = " << source_host_keys_state.size() << ") : " << std::endl;
            display_param("Source data:     ", source_host_keys_state_copy, n);
            display_param("Actual results:  ", host_keys, n);
            display_param("Expected results:", source_host_keys_state, n);

#endif // DUMP_CHECK_RESULTS
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last,
               Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        // copy source host keys state into temp buf
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        const std::vector<ValT> source_host_keys_state(host_keys.get(), host_keys.get() + n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        if (n == 1636)
            oneapi::dpl::exclusive_scan(new_policy, keys_first, keys_last, keys_first, kInitValue);
        else
            oneapi::dpl::exclusive_scan(new_policy, keys_first, keys_last, keys_first, kInitValue);
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n);

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::exclusive_scan(new_policy2, keys_first, keys_last, keys_first, kInitValue, DefaultBinaryOp<KeyT>());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n, DefaultBinaryOp<KeyT>());

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), n);
        update_data(host_keys);

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::exclusive_scan(new_policy3, keys_first, keys_last, keys_first, kInitValue, BinaryOperation());
        exec.queue().wait_and_throw();

        retrieve_data(host_keys);
        check_values(source_host_keys_state, host_keys.get(), n, BinaryOperation());
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, n);

        // copy source host keys state into temp buf
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        const std::vector<ValT> source_host_keys_state(keys_first, keys_first + n);

        auto res1 = oneapi::dpl::exclusive_scan(exec, keys_first, keys_last, keys_first, kInitValue);
        check_values(source_host_keys_state, keys_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, n);
        auto res2 = oneapi::dpl::exclusive_scan(exec, keys_first, keys_last, keys_first, DefaultBinaryOp<KeyT>());
        check_values(source_host_keys_state, keys_first, n, DefaultBinaryOp<KeyT>());

        // call algorithm with addition operator
        initialize_data(keys_first, n);
        auto res3 = oneapi::dpl::exclusive_scan(exec, keys_first, keys_last, keys_first, kInitValue, BinaryOperation());
        check_values(source_host_keys_state, keys_first, n, BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value, void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
    }
};

int
main()
{
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);

#if TEST_DPCPP_BACKEND_PRESENT

    using ValueType = int;  // ::std::uint64_t;
    using BinaryOperation = ::std::plus<ValueType>;

    // Run tests for USM shared memory
    test2buffers<sycl::usm::alloc::shared, test_inclusive_scan<ValueType, BinaryOperation>>();
    // Run tests for USM device memory
    test2buffers<sycl::usm::alloc::device, test_inclusive_scan<ValueType, BinaryOperation>>();

    // Run tests for USM shared memory
    test1buffer<sycl::usm::alloc::shared, test_inclusive_scan_inplace<ValueType, BinaryOperation>>();
    // Run tests for USM device memory
    test1buffer<sycl::usm::alloc::device, test_inclusive_scan_inplace<ValueType, BinaryOperation>>();

    // Run tests for USM shared memory
    test1buffer<sycl::usm::alloc::shared, test_exclusive_scan_inplace<ValueType, BinaryOperation>>();
    // Run tests for USM device memory
    test1buffer<sycl::usm::alloc::device, test_exclusive_scan_inplace<ValueType, BinaryOperation>>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
