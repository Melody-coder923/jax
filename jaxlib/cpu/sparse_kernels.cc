/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/cpu/sparse_kernels.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "xla/ffi/api/ffi.h"

namespace jax {

template <typename ElementType, typename StorageType>
using SparseMatrixType =
    Eigen::SparseMatrix<ElementType, Eigen::RowMajor, StorageType>;
template <typename ElementType>
using DenseMatrixType =
    Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename MatrixT>
using InputMap = Eigen::Map<const MatrixT, Eigen::Aligned32>;
template <typename MatrixT>
using OutputMap = Eigen::Map<MatrixT, Eigen::Aligned32>;

template <typename ElementType, typename StorageType>
static ::xla::ffi::Future CsrSparseDenseKernelImpl(
    const InputMap<SparseMatrixType<ElementType, StorageType>>& lhs_matrix,
    const InputMap<DenseMatrixType<ElementType>>& rhs_matrix,
    OutputMap<DenseMatrixType<ElementType>>& out_matrix,
    ::xla::ffi::ThreadPool& thread_pool) {
  // Rule of thumb to give each task at least 100k cycles to hide the cost of
  // task scheduling.
  // TODO(willfroom) Do we want to make this configurable?
  constexpr int64_t kTargetCyclesPerTask = 100'000;
  // Based on AVX (CPI 0.5 -> 2 IPC)
  constexpr int64_t kScalarProductsPerCycle = 2 * 32 / sizeof(ElementType);
  constexpr int64_t kTaskSize = kTargetCyclesPerTask * kScalarProductsPerCycle;

  if (lhs_matrix.nonZeros() * rhs_matrix.cols() <= kTaskSize ||
      thread_pool.num_threads() == 0) {
    out_matrix.noalias() = lhs_matrix * rhs_matrix;

    ::xla::ffi::Promise promise;
    promise.SetAvailable();
    return ::xla::ffi::Future(promise);
  } else {
    std::vector<int64_t> batch_sizes;
    {
      int64_t running_batch_nnz = 0;
      int64_t running_number_rows = 0;
      for (int row = 0; row < lhs_matrix.rows(); ++row) {
        int64_t row_nnz = lhs_matrix.outerIndexPtr()[row + 1] -
                          lhs_matrix.outerIndexPtr()[row];
        // If there is no non-zero elements in a row the task still needs to
        // write out a zero row we give each row a non-zero contribution to
        // avoid the pathological case of a task having to write many rows where
        // there is a large block of zero inputs.
        running_batch_nnz += std::max(row_nnz, static_cast<int64_t>(1));
        running_number_rows++;
        if (running_batch_nnz * rhs_matrix.cols() > kTaskSize) {
          batch_sizes.push_back(running_number_rows);
          running_batch_nnz = 0;
          running_number_rows = 0;
        } else if (row == lhs_matrix.rows() - 1 && running_number_rows > 0) {
          batch_sizes.push_back(running_number_rows);
        }
      }
    }

    ::xla::ffi::CountDownPromise promise(batch_sizes.size());
    ::xla::ffi::Future future(promise);
    int64_t batch_start = 0;
    for (int64_t size : batch_sizes) {
      thread_pool.Schedule([out_matrix, lhs_matrix, rhs_matrix, batch_start,
                            size, promise]() mutable {
        out_matrix.middleRows(batch_start, size).noalias() =
            lhs_matrix.middleRows(batch_start, size) * rhs_matrix;
        promise.CountDown();
      });
      batch_start += size;
    }
    return future;
  }
}

template <typename ElementType, typename StorageType>
static ::xla::ffi::Future CsrSparseDenseKernelTypedDispatch(
    ::xla::ffi::AnyBuffer lhs_data, ::xla::ffi::AnyBuffer lhs_outer_indicies,
    ::xla::ffi::AnyBuffer lhs_inner_indicies, ::xla::ffi::AnyBuffer rhs,
    ::xla::ffi::Result<::xla::ffi::AnyBuffer> out,
    ::xla::ffi::ThreadPool thread_pool) {
  ::xla::ffi::Span<const int64_t> rhs_shape = rhs.dimensions();
  ::xla::ffi::Span<const int64_t> out_shape = out->dimensions();

  InputMap<SparseMatrixType<ElementType, StorageType>> lhs_matrix(
      out_shape[0], rhs_shape[0], lhs_data.element_count(),
      lhs_outer_indicies.reinterpret_data<StorageType>(),
      lhs_inner_indicies.reinterpret_data<StorageType>(),
      lhs_data.reinterpret_data<ElementType>());

  InputMap<DenseMatrixType<ElementType>> rhs_matrix(
      rhs.reinterpret_data<ElementType>(), rhs_shape[0],
      rhs_shape.size() > 1 ? rhs_shape[1] : 1);
  OutputMap<DenseMatrixType<ElementType>> out_matrix(
      out->reinterpret_data<ElementType>(), lhs_matrix.rows(),
      rhs_matrix.cols());

  return CsrSparseDenseKernelImpl<ElementType, StorageType>(
      lhs_matrix, rhs_matrix, out_matrix, thread_pool);
}

template <typename ElementType>
static ::xla::ffi::Future CsrSparseDenseKernelTypedDispatch(
    ::xla::ffi::AnyBuffer lhs_data, ::xla::ffi::AnyBuffer lhs_outer_indicies,
    ::xla::ffi::AnyBuffer lhs_inner_indicies, ::xla::ffi::AnyBuffer rhs,
    ::xla::ffi::Result<::xla::ffi::AnyBuffer> out,
    ::xla::ffi::ThreadPool thread_pool) {
  if (lhs_outer_indicies.element_type() != lhs_inner_indicies.element_type()) {
    ::xla::ffi::Promise promise;
    promise.SetError(::xla::ffi::Error(::xla::ffi::ErrorCode::kInvalidArgument,
                                       "Sparse index type mismatch"));
    return ::xla::ffi::Future(promise);
  }

  switch (lhs_outer_indicies.element_type()) {
    case ::xla::ffi::DataType::S32:
      return CsrSparseDenseKernelTypedDispatch<ElementType, int32_t>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::S64:
      return CsrSparseDenseKernelTypedDispatch<ElementType, int64_t>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    default:
      ::xla::ffi::Promise promise;
      promise.SetError(::xla::ffi::Error(
          ::xla::ffi::ErrorCode::kInvalidArgument, "Invalid index data type"));
      return ::xla::ffi::Future(promise);
  }
}

static ::xla::ffi::Future CsrSparseDenseKernelDispatch(
    ::xla::ffi::AnyBuffer lhs_data, ::xla::ffi::AnyBuffer lhs_outer_indicies,
    ::xla::ffi::AnyBuffer lhs_inner_indicies, ::xla::ffi::AnyBuffer rhs,
    ::xla::ffi::Result<::xla::ffi::AnyBuffer> out,
    ::xla::ffi::ThreadPool thread_pool) {
  if (lhs_data.element_type() != rhs.element_type() ||
      lhs_data.element_type() != out->element_type()) {
    ::xla::ffi::Promise promise;
    promise.SetError(::xla::ffi::Error(::xla::ffi::ErrorCode::kInvalidArgument,
                                       "Element type mismatch"));
    return ::xla::ffi::Future(promise);
  }

  switch (lhs_data.element_type()) {
    case ::xla::ffi::DataType::S32:
      return CsrSparseDenseKernelTypedDispatch<int32_t>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::S64:
      return CsrSparseDenseKernelTypedDispatch<int64_t>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::F32:
      return CsrSparseDenseKernelTypedDispatch<float>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::F64:
      return CsrSparseDenseKernelTypedDispatch<double>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::C64:
      return CsrSparseDenseKernelTypedDispatch<std::complex<float>>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    case ::xla::ffi::DataType::C128:
      return CsrSparseDenseKernelTypedDispatch<std::complex<double>>(
          lhs_data, lhs_outer_indicies, lhs_inner_indicies, rhs, out,
          thread_pool);
    default:
      ::xla::ffi::Promise promise;
      promise.SetError(::xla::ffi::Error(
          ::xla::ffi::ErrorCode::kInvalidArgument, "Invalid data type"));
      return ::xla::ffi::Future(promise);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_csr_sparse_dense_ffi, CsrSparseDenseKernelDispatch,
    (::xla::ffi::Ffi::Bind()
         .Arg<::xla::ffi::AnyBuffer>(/*lhs_data*/)
         .Arg<::xla::ffi::AnyBuffer>(
             /*lhs_outer_indicies*/)
         .Arg<::xla::ffi::AnyBuffer>(
             /*lhs_inner_indicies*/)
         .Arg<::xla::ffi::AnyBuffer>(/*rhs*/)
         .Ret<::xla::ffi::AnyBuffer>(/*out*/)
         .Ctx<::xla::ffi::ThreadPool>(/*thread_pool*/)));

}  // namespace jax
