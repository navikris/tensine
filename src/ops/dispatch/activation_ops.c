#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/activation_ops.h"
#include "tensine/ops/kernels/activation_dispatch.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>


TsTensor* ts_softmax(const TsTensor* input, size_t dim) {
    assert(input);
    assert(dim < input->ndim);

    TsTensor* output = ts_tensor_empty_like(input);
    if (!output) return NULL;

    if (dim != (input->ndim - 1)) {
        TsTensor* permuted_input = ts_tensor_clone(input);
        size_t* permute_order = (size_t*)malloc(input->ndim * sizeof(size_t));
        for (size_t c_dim = 0; c_dim < input->ndim; ++c_dim) {
            permute_order[c_dim] = c_dim;
        }
        permute_order[input->ndim - 1] = dim;
        permute_order[dim] = input->ndim - 1;
        permuted_input = ts_tensor_permute(permuted_input, permute_order);
        permuted_input = ts_tensor_to_contiguous(permuted_input);

        memcpy(output->shape, permuted_input->shape, input->ndim * sizeof(size_t));
        ts_softmax_dispatch(permuted_input, output);
        ts_tensor_free(permuted_input);
        
        output = ts_tensor_permute(output, permute_order);
    } else {
        ts_softmax_dispatch(input, output);
    }
    return output;
}
