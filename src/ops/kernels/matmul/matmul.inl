#define CONCAT_NAME(a, b, c) CONCAT_NAME_IMPL(a, b, c)
#define CONCAT_NAME_IMPL(a, b, c) a##b##c


void CONCAT_NAME(matmul_, NAME, _cpu)(
    const T* restrict input_1,
    const T* restrict input_2,
    T* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
) {
    /* 
    Considerations:
    1. All higher dim tensors to be flattened to [Batch, H, W]
    2. Only tensors are of contiguous layout 
    */
    size_t numel_to_macc = input_1_shape[ndim - 1];
    size_t* input_1_strides = _get_strides(input_1_shape, ndim);
    size_t* input_2_strides = _get_strides(input_2_shape, ndim);
    size_t* output_strides = _get_strides(output_shape, ndim);
    size_t* out_dims = (size_t*)malloc(ndim * sizeof(size_t));
    memset(out_dims, 0, ndim * sizeof(size_t));

    size_t batch_offset_input_1 = 0;
    size_t batch_offset_input_2 = 0;
    size_t batch_offset_output = 0;
    while(1) {
        for (size_t dim = (ndim - 2); dim-- > 0;) {
            if (dim > ndim) {
                break; // for case where ndim <= 2 to handle underflow
            }
            if (out_dims[dim] <= output_shape[dim]) {
                memset(out_dims + dim + 1, 0, (ndim - dim - 1) * sizeof(size_t));
                batch_offset_input_1 += out_dims[dim] * input_1_strides[dim];
                batch_offset_input_2 += out_dims[dim] * input_2_strides[dim];
                batch_offset_output += out_dims[dim] * output_strides[dim];
                out_dims[dim] += 1;
                break;
            }
        }

        for (size_t row = 0; row < output_shape[ndim - 2]; ++row) {
            for (size_t col = 0; col < output_shape[ndim - 1]; ++col) {
                size_t input_1_offset = batch_offset_input_1 + row * input_1_shape[ndim - 1];
                size_t input_2_offset = batch_offset_input_2 + col;
                size_t output_offset = batch_offset_output + row * output_shape[ndim - 1] + col;
                CONCAT_NAME(vector_mac_, NAME, _cpu)(
                    (input_1 + input_1_offset),
                    (input_2 + input_2_offset),
                    (output + output_offset),
                    input_1_strides[ndim - 1],
                    input_2_strides[ndim - 2],
                    numel_to_macc
                );
            }
        }
        if (ndim == 2) {
            break;
        }

        if (out_dims[0] >= output_shape[0]) {
            break;
        }
    }
}
