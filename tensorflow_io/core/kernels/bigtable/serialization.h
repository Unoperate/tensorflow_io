/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "absl/memory/memory.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "rpc/rpc.h" /* xdr is a sub-library of rpc */
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_resource_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_range.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace data {
namespace {

std::string FloatToBytes(float v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_float(&xdrs, &v)) {
    throw std::runtime_error("Error writing float to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

float BytesToFloat(std::string const& s) {
  float v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_float(&xdrs, &v)) {
    throw std::runtime_error("Error reading float from byte array.");
  }
  return v;
}

std::string DoubleToBytes(double v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_double(&xdrs, &v)) {
    throw std::runtime_error("Error writing double to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

double BytesToDouble(std::string const& s) {
  double v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_double(&xdrs, &v)) {
    throw std::runtime_error("Error reading double from byte array.");
  }
  return v;
}

std::string Int64ToBytes(int64_t v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    throw std::runtime_error("Error writing int64 to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

int64_t BytesToInt64(std::string const& s) {
  int64_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    throw std::runtime_error("Error reading int64 from byte array.");
  }
  return v;
}

std::string Int32ToBytes(int32_t v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_int32_t(&xdrs, &v)) {
    throw std::runtime_error("Error writing int32 to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

int32_t BytesToInt32(std::string const& s) {
  int32_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int32_t(&xdrs, &v)) {
    throw std::runtime_error("Error reading int64 from byte array.");
  }
  return v;
}

std::string BoolToBytes(bool_t v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_bool(&xdrs, &v)) {
    throw std::runtime_error("Error writing bool to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

bool_t BytesToBool(std::string const& s) {
  bool_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_bool(&xdrs, &v)) {
    throw std::runtime_error("Error reading bool from byte array.");
  }
  return v;
}

Status PutCellValueInTensor(Tensor& tensor, size_t index, DataType cell_type,
                            cbt::Cell const& cell) {
  switch (cell_type) {
    case DT_STRING: {
      auto tensor_data = tensor.tensor<tstring, 1>();
      tensor_data(index) = std::string(cell.value());
    } break;
    case DT_BOOL: {
      auto tensor_data = tensor.tensor<bool, 1>();
      tensor_data(index) = BytesToBool(cell.value());
    } break;
    case DT_INT32: {
      auto tensor_data = tensor.tensor<int32_t, 1>();
      tensor_data(index) = BytesToInt32(cell.value());
    } break;
    case DT_INT64: {
      auto tensor_data = tensor.tensor<int64_t, 1>();
      tensor_data(index) = BytesToInt64(cell.value());
    } break;
    case DT_FLOAT: {
      auto tensor_data = tensor.tensor<float, 1>();
      tensor_data(index) = BytesToFloat(cell.value());
    } break;
    case DT_DOUBLE: {
      auto tensor_data = tensor.tensor<double, 1>();
      tensor_data(index) = BytesToDouble(cell.value());
    } break;
    default:
      return errors::Unimplemented("Data type not supported.");
  }
  return Status::OK();
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

#endif /* SERIALIZATION_H */