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

// This file contains SYCL specific macros and abstractions
// to support different versions of SYCL and to simplify its interfaces
//
// Functionality, required for the tests, is copied from sycl_defs.h and thus is not directly used
// for compatibility with different versions of the library

#ifndef _UTILS_SYCL_DEFS_H
#define _UTILS_SYCL_DEFS_H

#include <CL/sycl.hpp>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define __LIBSYCL_VERSION                                                                                          \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define __LIBSYCL_VERSION 0
#endif

#if _ONEDPL_FPGA_DEVICE
#    if __LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

#define _ONEDPL_NO_INIT_PRESENT (__LIBSYCL_VERSION >= 50300)

namespace TestUtils
{
using __no_init =
#if _ONEDPL_NO_INIT_PRESENT
    sycl::property::no_init;
#else
    sycl::property::noinit;
#endif

#if _ONEDPL_FPGA_DEVICE
#    if __LIBSYCL_VERSION >= 50300
using __fpga_emulator_selector = sycl::ext::intel::fpga_emulator_selector;
using __fpga_selector = sycl::ext::intel::fpga_selector;
#    else
using __fpga_emulator_selector = sycl::INTEL::fpga_emulator_selector;
using __fpga_selector = sycl::INTEL::fpga_selector;
#    endif
#endif // _ONEDPL_FPGA_DEVICE

} // TestUtils

#endif //  _UTILS_SYCL_DEFS_H
