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

struct Flip
{
    std::int32_t val;
    Flip(std::int32_t y) : val(y) {}
    template <typename T>
    T
    operator()(const T& x) const
    {
        return val - x;
    }
};
struct Plus
{
    template <typename T, typename U>
    T
    operator()(const T x, const U y) const
    {
        return x + y;
    }
};

struct Inc
{
    template <typename T>
    void
    operator()(T& x) const
    {
        ++x;
    }
};

template <typename T>
struct Generator_count
{
    T def_val;
    Generator_count(const T& val) : def_val(val) {}
    T
    operator()() const
    {
        return def_val;
    }
    T
    default_value() const
    {
        return def_val;
    }
};

// created just to check destroy and destroy_n correctness
template <typename T>
struct SyclTypeWrapper
{
    T __value;

    explicit SyclTypeWrapper(const T& value = T{4}) : __value(value) {}
    ~SyclTypeWrapper() { __value = -2; }
    bool
    operator==(const SyclTypeWrapper& other) const
    {
        return __value == other.__value;
    }
};

// this wrapper is needed to take into account not only kernel name,
// but also other types (for example, iterator's value type)
template<typename... T>
struct policy_name_wrapper{};

using namespace oneapi::dpl::execution;

DEFINE_TEST(test_uninitialized_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_copy");
    }
};

DEFINE_TEST(test_uninitialized_copy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_copy_n)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{0});
        update_data(host_keys, host_vals);

        ::std::uninitialized_copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_copy_n");
    }
};

DEFINE_TEST(test_uninitialized_move)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_move)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_move");
    }
};

DEFINE_TEST(test_uninitialized_move_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_move_n)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_move_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from uninitialized_move_n");
    }
};

DEFINE_TEST(test_uninitialized_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                                  value);

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2),
                                 value),
                    "wrong effect from uninitialized_fill");
    }
};

DEFINE_TEST(test_uninitialized_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from uninitialized_fill_n");
    }
};

DEFINE_TEST(test_uninitialized_default_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                             first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), exp_value),
                    "wrong effect from uninitialized_default_construct");
    }
};

DEFINE_TEST(test_uninitialized_default_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, exp_value),
                    "wrong effect from uninitialized_default_construct_n");
    }
};

DEFINE_TEST(test_uninitialized_value_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                           first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), T1{}),
                    "wrong effect from uninitialized_value_construct");
    }
};

DEFINE_TEST(test_uninitialized_value_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, T1{}),
                    "wrong effect from uninitialized_value_construct_n");
    }
};

DEFINE_TEST(test_destroy)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1 + (n / 3),
                       first1 + (n / 2));
        if (!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from destroy");
    }
};

DEFINE_TEST(test_destroy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy_n(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1, n);
        if(!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value),
                    "wrong effect from destroy_n");
    }
};

DEFINE_TEST(test_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), value);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value), "wrong effect from fill");
    }
};

DEFINE_TEST(test_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1), "wrong effect from fill_n");
    }
};

DEFINE_TEST(test_generate)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                      Generator_count<T1>(value));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from generate");
    }
};

DEFINE_TEST(test_generate_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Generator_count<T1>(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from generate_n");
    }
};

DEFINE_TEST(test_for_each)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value - 1);
        host_keys.update_data();

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), Inc());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        // We call due to SYCL 1.2.1: 4.7.2.3.
        // If the host memory is modified by the host,
        // or mapped to another buffer or image during the lifetime of this buffer,
        // then the results are undefined
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value), "wrong effect from for_each");
    }
};

DEFINE_TEST(test_for_each_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::for_each_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Inc());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from for_each_n");
    }
};

DEFINE_TEST(test_transform_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_unary)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, value + 1);
        update_data(host_keys, host_vals);

        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + n / 2, last1, first2 + n / 2, Flip(7));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n / 2, value + 1),
                    "wrong effect from transform_unary (1)");
        EXPECT_TRUE(check_values(host_vals.get() + n / 2, host_vals.get() + n, T1(5)),
                    "wrong effect from transform_unary (2)");
    }
};

DEFINE_TEST(test_transform_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_binary)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(3);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first1, first2, Plus());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, T1(6)),
                    "wrong effect from transform_binary");
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_for_each<ValueType>>();
    PRINT_DEBUG("test_for_each_n");
    test1buffer<alloc_type, test_for_each_n<ValueType>>();
    PRINT_DEBUG("test_fill");
    test1buffer<alloc_type, test_fill<ValueType>>();
    PRINT_DEBUG("test_fill_n");
    test1buffer<alloc_type, test_fill_n<ValueType>>();
    PRINT_DEBUG("test_generate");
    test1buffer<alloc_type, test_generate<ValueType>>();
    PRINT_DEBUG("test_generate_n");
    test1buffer<alloc_type, test_generate_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill");
    test1buffer<alloc_type, test_uninitialized_fill<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill_n");
    test1buffer<alloc_type, test_uninitialized_fill_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_default_construct");
    test1buffer<alloc_type, test_uninitialized_default_construct<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_default_construct_n");
    test1buffer<alloc_type, test_uninitialized_default_construct_n<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_value_construct");
    test1buffer<alloc_type, test_uninitialized_value_construct<ValueType>>();
    PRINT_DEBUG("test_uninitialized_value_construct_n");
    test1buffer<alloc_type, test_uninitialized_value_construct_n<ValueType>>();
    PRINT_DEBUG("test_destroy");
    test1buffer<alloc_type, test_destroy<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_destroy_n");
    test1buffer<alloc_type, test_destroy_n<SyclTypeWrapper<ValueType>>>();
    test1buffer<alloc_type, test_destroy_n<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_transform_unary");
    test2buffers<alloc_type, test_transform_unary<ValueType>>();
    PRINT_DEBUG("test_transform_binary");
    test2buffers<alloc_type, test_transform_binary<ValueType>>();
    PRINT_DEBUG("test_uninitialized_copy");
    test2buffers<alloc_type, test_uninitialized_copy<ValueType>>();
    PRINT_DEBUG("test_uninitialized_copy_n");
    test2buffers<alloc_type, test_uninitialized_copy_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_move");
    test2buffers<alloc_type, test_uninitialized_move<ValueType>>();
    PRINT_DEBUG("test_uninitialized_move_n");
    test2buffers<alloc_type, test_uninitialized_move_n<ValueType>>();
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
