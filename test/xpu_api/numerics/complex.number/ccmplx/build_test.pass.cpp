#define _GLIBCXX_USE_TBB_PAR_BACKEND 0 // libstdc++10

#include <cassert>

#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <algorithm>
#include <vector>
#include <iostream>

#include <numeric> // std::inclusive_scan, exclusive_scan
#include <functional>

int
main()
{
    std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};

    // Setup host inputs
    std::vector<int> incl_input_host = v;
    std::vector<int> excl_input_host = v;

    // Setup device inputs
    sycl::queue syclQue(sycl::gpu_selector{});
    int* incl_input_dev = sycl::malloc_device<int>(10, syclQue);
    int* excl_input_dev = sycl::malloc_device<int>(10, syclQue);

    syclQue.memcpy(incl_input_dev, v.data(), v.size() * sizeof(int)).wait();
    syclQue.memcpy(excl_input_dev, v.data(), v.size() * sizeof(int)).wait();

    // Inclusive scan (in-place works)
    std::inclusive_scan(incl_input_host.begin(), incl_input_host.end(), incl_input_host.begin());
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(syclQue), incl_input_dev,
                                incl_input_dev + v.size(), incl_input_dev);
    int* incl_result_host = new int[v.size()];
    syclQue.memcpy(incl_result_host, incl_input_dev, v.size() * sizeof(int)).wait();

    for (int i = 0; i < v.size(); i++)
    {
        assert(incl_input_host[i] == incl_result_host[i]);
    }
    delete[] incl_result_host;
    sycl::free(incl_input_dev, syclQue);

    // Exclusive scan (in-place, incorrect results)
    std::exclusive_scan(excl_input_host.begin(), excl_input_host.end(), excl_input_host.begin(), 0);
    oneapi::dpl::exclusive_scan(oneapi::dpl::execution::make_device_policy(syclQue), excl_input_dev,
                                excl_input_dev + v.size(), excl_input_dev, 0);
    int* excl_result_host = new int[v.size()];
    syclQue.memcpy(excl_result_host, excl_input_dev, v.size() * sizeof(int)).wait();

    for (int i = 0; i < v.size(); i++)
    {
        assert(excl_input_host[i] == excl_result_host[i]);
    }
    delete[] excl_result_host;
    sycl::free(excl_input_dev, syclQue);

    return 0;
}
