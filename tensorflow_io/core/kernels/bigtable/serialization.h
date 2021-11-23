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
==============================================================================

Byte representation of basic types may depend on language and architecture, thus
we need something that is invariant and consistent with HBase. XDR library seems
to satisfy those constraints.
*/

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "google/cloud/bigtable/table.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace io {

Status PutCellValueInTensor(Tensor& tensor, size_t index,
                                   DataType cell_type,
                                   google::cloud::bigtable::Cell const& cell);

}  // namespace io
}  // namespace tensorflow

#endif /* SERIALIZATION_H */