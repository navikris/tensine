#define CONCAT_NAME(a, b, c) CONCAT_NAME_IMPL(a, b, c)
#define CONCAT_NAME_IMPL(a, b, c) a##b##c


void CONCAT_NAME(maxpool2d_, NAME, _cpu)(
    const T* restrict input,
    T* restrict output,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding,
    size_t ndim
) {
    /* 
    Considerations:
    1. Tensors are of contiguous layout 
    3. Symmetric padding
    */
    size_t* input_strides = get_strides(input_shape, ndim);
    size_t* output_strides = get_strides(output_shape, ndim);

    for (size_t batch = 0; batch < output_shape[ndim - 4]; ++batch) {// batch iterator
        for (size_t out_h = 0; out_h < output_shape[ndim - 3]; ++out_h) {// height iterator
            for (size_t out_w = 0; out_w < output_shape[ndim - 2]; ++out_w) {// width iterator
                for (size_t out_ch = 0; out_ch < output_shape[ndim - 1]; ++out_ch) {
                    size_t output_index = batch * output_strides[ndim - 4] \
                                            + out_h * output_strides[ndim - 3] \
                                            + out_w * output_strides[ndim - 2] \
                                            + out_ch;
    
                    size_t pad_h = (out_h == 1) ? padding[0] : 0;
                    size_t pad_w = (out_w == 1) ? padding[1] : 0;
                    size_t input_index = batch * input_strides[ndim - 4] \
                                            + ((out_h * stride[0]) - pad_h) * input_strides[ndim - 3] \
                                            + ((out_w * stride[1]) - pad_w) * input_strides[ndim - 2] \
                                            + out_ch;
    
                    T pooled_out = input[input_index];
                    for (size_t k_h = 0; k_h < kernel_size[0]; ++k_h) {
                        if ((out_h == (output_shape[ndim - 3] - 1)) && ((kernel_size[0] - k_h - 1) <= padding[0])) {
                            break;
                        }
                        for (size_t k_w = 0; k_w < kernel_size[1]; ++k_w) {
                            if ((out_w == (output_shape[ndim - 2] - 1)) && ((kernel_size[1] - k_w - 1) <= padding[1])) {
                                break;
                            }
                            size_t comp_index = input_index \
                                            + k_h * input_strides[ndim - 3] \
                                            + k_w * input_strides[ndim - 2];
                            if (input[comp_index] > pooled_out) {
                                pooled_out = input[comp_index];
                            }
                        }
                    }
                    output[output_index] = pooled_out;
                }
            }
        }
    }
}
