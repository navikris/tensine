#define CONCAT_NAME(a, b, c) CONCAT_NAME_IMPL(a, b, c)
#define CONCAT_NAME_IMPL(a, b, c) a##b##c


void CONCAT_NAME(softmax_, NAME, _cpu)(
    const T* restrict input,
    T* restrict output,
    const size_t* input_shape,
    size_t ndim
) {
    /* 
    Considerations:
    1. All higher dim tensors to be flattened to [Batch, C]
    2. Tensors are of contiguous layout
    3. Only softmax in last dim is supported
    */
    size_t* input_strides = get_strides(input_shape, ndim);
    size_t* out_dims = (size_t*)malloc(ndim * sizeof(size_t));
    memset(out_dims, 0, ndim * sizeof(size_t));
   
    while(1) { 
        size_t batch_offset = 0;
        for (size_t dim = 0; dim < ndim - 1; ++dim) {
            batch_offset += out_dims[dim] * input_strides[dim];
        }

        T max = input[batch_offset];
        for (size_t col = 1; col < input_shape[ndim - 1]; ++col) {
            if (max < input[batch_offset + col]) {
                max = input[batch_offset + col];
            }
        }

        for (size_t col = 0; col < input_shape[ndim - 1]; ++col) {
            output[batch_offset + col] = exp(input[batch_offset + col] - max);
        }

        T exp_sum = output[batch_offset];
        for (size_t col = 1; col < input_shape[ndim - 1]; ++col) {
            exp_sum += output[batch_offset + col];
        }

        for (size_t col = 0; col < input_shape[ndim - 1]; ++col) {
            output[batch_offset + col] = output[batch_offset + col] / exp_sum;
        }

        if (ndim == 1) {
            break;
        }

        for (size_t dim = (ndim - 1); dim-- > 0;) {
            if (dim > ndim) {
                break; // for case where ndim = 1 to handle underflow
            }
            out_dims[dim] += 1;
            if (out_dims[dim] < input_shape[dim]) {
                memset(out_dims + dim + 1, 0, (ndim - dim - 1) * sizeof(size_t));
                break;
            }
        }
        if (out_dims[0] >= input_shape[0]) {
            break;
        }
    }
}
