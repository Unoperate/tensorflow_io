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

#ifndef BIGTABLE_ROW_RANGE_H
#define BIGTABLE_ROW_RANGE_H

#include "absl/memory/memory.h"
#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow_io/core/kernels/bigtable/bigtable_resource_kernel.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace data {
namespace {

class BigtableRowRangeResource : public ResourceBase {
 public:
  explicit BigtableRowRangeResource(cbt::RowRange row_range)
      : row_range_(std::move(row_range)) {
    VLOG(1) << "BigtableRowsetResource ctor";
  }

  ~BigtableRowRangeResource() { VLOG(1) << "BigtableRowsetResource dtor"; }

  std::string ToString() const {
    std::string res;
    google::protobuf::TextFormat::PrintToString(row_range_.as_proto(), &res);
    return res;
  }

  cbt::RowRange& RowRange() { return row_range_; }

  string DebugString() const override {
    return "BigtableRowRangeResource:{" + ToString() + "}";
  }

 private:
  cbt::RowRange row_range_;
};

class BigtableEmptyRowRangeOp
    : public OpKernelCreatingResource<BigtableRowRangeResource> {
 public:
  explicit BigtableEmptyRowRangeOp(OpKernelConstruction* ctx)
      : OpKernelCreatingResource<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtableEmptyRowRangeOp ctor ";
  }

 private:
  Status CreateResource(BigtableRowRangeResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new BigtableRowRangeResource(cbt::RowRange::Empty());
    return Status::OK();
  }

 private:
  mutable mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableEmptyRowRange").Device(DEVICE_CPU),
                        BigtableEmptyRowRangeOp);

class BigtableRowRangeOp
    : public OpKernelCreatingResource<BigtableRowRangeResource> {
 public:
  explicit BigtableRowRangeOp(OpKernelConstruction* ctx)
      : OpKernelCreatingResource<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtableRowRangeOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_row_key", &left_row_key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_open", &left_open_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_row_key", &right_row_key_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_open", &right_open_));
  }

 private:
  Status CreateResource(BigtableRowRangeResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    VLOG(1) << "BigtableRowRangeOp constructing row_range:"
            << (left_open_ ? "(" : "[") << left_row_key_ << ":"
            << right_row_key_ << (right_open_ ? ")" : "]");

    // both empty - infinite
    if (left_row_key_.empty() && right_row_key_.empty()) {
      *resource = new BigtableRowRangeResource(cbt::RowRange::InfiniteRange());
      return Status::OK();
    }

    // open
    if (left_open_ && right_open_) {
      *resource = new BigtableRowRangeResource(
          cbt::RowRange::Open(left_row_key_, right_row_key_));
      return Status::OK();
    }
    // closed
    if (!left_open_ && !right_open_) {
      *resource = new BigtableRowRangeResource(
          cbt::RowRange::Closed(left_row_key_, right_row_key_));
      return Status::OK();
    }
    // right_open
    if (!left_open_ && right_open_) {
      *resource = new BigtableRowRangeResource(
          cbt::RowRange::RightOpen(left_row_key_, right_row_key_));
      return Status::OK();
    }
    // left_open
    if (left_open_ && !right_open_) {
      *resource = new BigtableRowRangeResource(
          cbt::RowRange::LeftOpen(left_row_key_, right_row_key_));
      return Status::OK();
    }
    return errors::Internal("Reached impossible branch.");
  }

 private:
  mutable mutex mu_;
  std::string left_row_key_ TF_GUARDED_BY(mu_);
  bool left_open_ TF_GUARDED_BY(mu_);
  std::string right_row_key_ TF_GUARDED_BY(mu_);
  bool right_open_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowRange").Device(DEVICE_CPU),
                        BigtableRowRangeOp);

class BigtablePrintRowRangeOp : public OpKernel {
 public:
  explicit BigtablePrintRowRangeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableRowRangeResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }

 private:
  mutable mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintRowRange").Device(DEVICE_CPU),
                        BigtablePrintRowRangeOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow

#endif /* BIGTABLE_ROW_RANGE_H */
