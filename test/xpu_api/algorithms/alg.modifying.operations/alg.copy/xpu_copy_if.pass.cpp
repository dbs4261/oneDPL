//===-- xpu_copy_if.pass.cpp ----------------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"

#include <cassert>
#include <sycl.hpp>

struct Pred
{
    bool
    operator()(int i)
    {
        return i % 3 == 0;
    }
};

template <class Iter1, class Iter2>
class KernelTest;

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 1000;
    int ia[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};
    bool ret = true;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        sycl::buffer<int, 1> buffer2(ib, itemN);
        sycl::buffer<bool, 1> buffer3(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl::access::mode::read>(cgh);
            auto acc_arr2 = buffer2.get_access<sycl::access::mode::write>(cgh);
            auto ret_acc = buffer3.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest<InIter, OutIter>>([=]() {
                OutIter r = dpl::copy_if(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + N), OutIter(&acc_arr2[0]), Pred());
                ret_acc[0] &= (base(r) == &acc_arr2[0] + N / 3 + 1);
            });
        });
    }
    assert(ret);
    for (size_t idx = 0; idx < N; ++idx)
    {
        assert(ib[idx] % 3 == 0);
    }
}

int
main()
{
    sycl::queue deviceQueue;
    test<input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, output_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, int*>(deviceQueue);
    return 0;
}
