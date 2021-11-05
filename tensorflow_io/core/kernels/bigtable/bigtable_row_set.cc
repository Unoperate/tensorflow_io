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
#include "tensorflow_io/core/kernels/bigtable/bigtable_row_set.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {

class BigtableEmptyRowsetOp
    : public AbstractBigtableResourceOp<BigtableRowsetResource> {
 public:
  explicit BigtableEmptyRowsetOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowsetResource>(ctx) {
    VLOG(1) << "BigtableEmptyRowsetOp ctor ";
  }

 private:
  StatusOr<BigtableRowsetResource*> CreateResource()
       override {
    return new BigtableRowsetResource(cbt::RowSet());
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtableEmptyRowset").Device(DEVICE_CPU),
                        BigtableEmptyRowsetOp);

class BigtablePrintRowsetOp : public OpKernel {
 public:
  explicit BigtablePrintRowsetOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableRowsetResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintRowset").Device(DEVICE_CPU),
                        BigtablePrintRowsetOp);

class BigtableRowsetAppendRowOp : public OpKernel {
 public:
  explicit BigtableRowsetAppendRowOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("row_key", &row_key_));
  }

  void Compute(OpKernelContext* context) override {
    BigtableRowsetResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "resource", &resource));
    core::ScopedUnref unref(resource);

    resource->AppendRow(row_key_);
  }

 private:
  std::string row_key_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowsetAppendRow").Device(DEVICE_CPU),
                        BigtableRowsetAppendRowOp);

class BigtableRowsetAppendRowRangeOp : public OpKernel {
 public:
  explicit BigtableRowsetAppendRowRangeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    mutex_lock lock(mu_);
    BigtableRowsetResource* row_set_resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "row_set_resource",
                                                   &row_set_resource));
    core::ScopedUnref row_set_resource_unref(row_set_resource);

    BigtableRowRangeResource* row_range_resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "row_range_resource",
                                          &row_range_resource));
    core::ScopedUnref row_range_resource_unref(row_range_resource);

    row_set_resource->AppendRowRange(row_range_resource->row_range());
  }

 private:
  mutable mutex mu_;
  std::string row_key_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowsetAppendRowRange").Device(DEVICE_CPU),
                        BigtableRowsetAppendRowRangeOp);

class BigtablePrefixRowRangeOp
    : public AbstractBigtableResourceOp<BigtableRowRangeResource> {
 public:
  explicit BigtablePrefixRowRangeOp(OpKernelConstruction* ctx)
      : AbstractBigtableResourceOp<BigtableRowRangeResource>(ctx) {
    VLOG(1) << "BigtablePrefixRowRangeOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prefix_str", &prefix_str_));
  }

 private:
  StatusOr<BigtableRowRangeResource*> CreateResource()
       override {
    return new BigtableRowRangeResource(cbt::RowRange::Prefix(prefix_str_));
  }

 private:
  std::string prefix_str_;
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrefixRowRange").Device(DEVICE_CPU),
                        BigtablePrefixRowRangeOp);

class BigtableRowsetIntersectOp : public OpKernel {
 public:
  explicit BigtableRowsetIntersectOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    ResourceMgr* mgr = context->resource_manager();
    OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

    BigtableRowsetResource* row_set_resource;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "row_set_resource",
                                                   &row_set_resource));
    core::ScopedUnref row_set_resource_unref(row_set_resource);

    BigtableRowRangeResource* row_range_resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "row_range_resource",
                                          &row_range_resource));
    core::ScopedUnref row_range_resource_unref(row_range_resource);

    BigtableRowsetResource* result_resource;
    OP_REQUIRES_OK(
        context,
        mgr->LookupOrCreate<BigtableRowsetResource>(
            cinfo_.container(), cinfo_.name(), &result_resource,
            [this, row_set_resource, row_range_resource](
                BigtableRowsetResource** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              *ret = new BigtableRowsetResource(
                  row_set_resource->Intersect(row_range_resource->row_range()));
              return Status::OK();
            }));

    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, cinfo_.container(), cinfo_.name(),
                                TypeIndex::Make<BigtableRowsetResource>()));
  }

 protected:
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(Name("BigtableRowsetIntersect").Device(DEVICE_CPU),
                        BigtableRowsetIntersectOp);

}  // namespace io
}  // namespace tensorflow
