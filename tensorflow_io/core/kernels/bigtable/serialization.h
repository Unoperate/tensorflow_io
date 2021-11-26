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

#include "tensorflow/core/platform/statusor.h"
#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/tensor.h"
#include "rpc/xdr.h"

namespace tensorflow {
namespace io {

class Serializer {
    public:

    Serializer(){
        const char* var = std::getenv("TFIO_DONT_USE_XDR");
        VLOG(1) << "got env TFIO_DONT_USE_XDR=" << var;
        if(var && var[0] == '1'){
            VLOG(1) << "using custom implementation for serialization";
            use_xdr_ = false;
        } else {
            VLOG(1) << "using XDR for serialization";
            use_xdr_ = true;
        }
    }


    // Bigtable only stores values as byte buffers - except for int64 the server
    // side does not have any notion of types. Tensorflow, needs to store shorter
    // integers, floats, doubles, so we needed to decide on how. We chose to follow
    // what HBase does, since there is a path for migrating from HBase to Bigtable.
    // XDR seems to match what HBase does.
    Status PutCellValueInTensor(Tensor& tensor, size_t index, DataType cell_type,
                                google::cloud::bigtable::Cell const& cell) const;
    
    private:
    bool use_xdr_;
};


}  // namespace io
}  // namespace tensorflow

#endif /* SERIALIZATION_H */