#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;

class BigtableDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit BigtableDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    // Parse and validate any attrs that define the dataset using
    // `ctx->GetAttr()`, and store them in member variables.
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase** output) override {
    // Parse and validate any input tensors that define the dataset using
    // `ctx->input()` or the utility function
    // `ParseScalarArgument<T>(ctx, &arg)`.

    // Create the dataset object, passing any (already-validated) arguments from
    // attrs or input tensors.
    *output = new Dataset(ctx);
  }

 private:
  class Dataset : public tensorflow::data::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx)
        : tensorflow::data::DatasetBase(tensorflow::data::DatasetContext(ctx)) {
    }

    std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const {
      return std::unique_ptr<tensorflow::data::IteratorBase>(new Iterator(
          {this, tensorflow::strings::StrCat(prefix, "::BigtableDataset")}));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const tensorflow::DataTypeVector& output_dtypes() const override {
      static auto* const dtypes = new tensorflow::DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    std::string DebugString() const override {
      return "BigtableDatasetOp::Dataset";
    }

   protected:
    // Optional: Implementation of `GraphDef` serialization for this dataset.
    //
    // Implement this method if you want to be able to save and restore
    // instances of this dataset (and any iterators over it).
    Status AsGraphDefInternal(tensorflow::data::SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              tensorflow::Node** output) const {
      // Construct nodes to represent any of the input tensors from this
      // object's member variables using `b->AddScalar()` and `b->AddVector()`.

      return tensorflow::errors::Unimplemented(
          "%s does not support serialization", DebugString());
    }

    Status CheckExternalState() const override { return Status::OK(); }

   private:
    class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), i_(0) {}

      // Implementation of the reading logic.
      //
      // The example implementation in this file yields the string "MyReader!"
      // ten times. In general there are three cases:
      //
      // 1. If an element is successfully read, store it as one or more tensors
      //    in `*out_tensors`, set `*end_of_sequence = false` and return
      //    `Status::OK()`.
      // 2. If the end of input is reached, set `*end_of_sequence = true` and
      //    return `Status::OK()`.
      // 3. If an error occurs, return an error status using one of the helper
      //    functions from "tensorflow/core/lib/core/errors.h".
      Status GetNextInternal(tensorflow::data::IteratorContext* ctx,
                             std::vector<tensorflow::Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // NOTE: `GetNextInternal()` may be called concurrently, so it is
        // recommended that you protect the iterator state with a mutex.
        tensorflow::mutex_lock l(mu_);
        if (i_ < 10) {
          // Create a scalar string tensor and add it to the output.
          tensorflow::Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
          record_tensor.scalar<tensorflow::tstring>()() = "MyReader!";
          out_tensors->emplace_back(std::move(record_tensor));
          ++i_;
          *end_of_sequence = false;
        } else {
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     protected:
      Status SaveInternal(
          tensorflow::data::SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        return tensorflow::errors::Unimplemented("SaveInternal");
      }

      Status RestoreInternal(
          tensorflow::data::IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        return tensorflow::errors::Unimplemented(
            "Iterator does not support 'RestoreInternal')");
      }

     private:
      tensorflow::mutex mu_;
      tensorflow::int64 i_ GUARDED_BY(mu_);
    };
  };
};


// Register the kernel implementation for MyReaderDataset.
REGISTER_KERNEL_BUILDER(Name("BigtableDataset").Device(tensorflow::DEVICE_CPU),
                        BigtableDatasetOp);