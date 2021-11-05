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
#include "tensorflow_io/core/kernels/bigtable/bigtable_version_filters.h"

namespace cbt = ::google::cloud::bigtable;

namespace tensorflow {
namespace io {

class BigtableLatestFilterOp
    : public OpKernelCreatingResource<BigtableFilterResource> {
 public:
  explicit BigtableLatestFilterOp(OpKernelConstruction* ctx)
      : OpKernelCreatingResource<BigtableFilterResource>(ctx) {
    VLOG(1) << "BigtableLatestFilterOp ctor ";
  }

 private:
  Status CreateResource(BigtableFilterResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new BigtableFilterResource(cbt::Filter::Latest(1));
    return Status::OK();
  }

 private:
  mutable mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableLatestFilter").Device(DEVICE_CPU),
                        BigtableLatestFilterOp);
    
class BigtableTimestampRangeFilterOp
    : public OpKernelCreatingResource<BigtableFilterResource> {
 public:
  explicit BigtableTimestampRangeFilterOp(OpKernelConstruction* ctx)
      : OpKernelCreatingResource<BigtableFilterResource>(ctx) {
    VLOG(1) << "BigtableTimestampRangeFilterOp ctor ";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("start", &start_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end", &end_));
  }

 private:
  Status CreateResource(BigtableFilterResource** resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *resource = new BigtableFilterResource(cbt::Filter::TimestampRangeMicros(start_, end_));
    return Status::OK();
  }

 private:
  mutable mutex mu_;
  int64_t start_;
  int64_t end_;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTimestampRangeFilter").Device(DEVICE_CPU),
                        BigtableTimestampRangeFilterOp);

class BigtablePrintFilterOp : public OpKernel {
 public:
  explicit BigtablePrintFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    BigtableFilterResource* resource;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "filter", &resource));
    core::ScopedUnref unref(resource);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
    auto output_v = output_tensor->tensor<tstring, 1>();

    output_v(0) = resource->ToString();
  }

 private:
  mutable mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrintFilter").Device(DEVICE_CPU),
                        BigtablePrintFilterOp);

} // namespace io
} // namespace tensorflow
