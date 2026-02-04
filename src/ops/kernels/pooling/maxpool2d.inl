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
    size_t required_input_height = (output_shape[ndim - 3] - 1) * stride[0] + kernel_size[0];
    size_t required_input_width = (output_shape[ndim - 2] - 1) * stride[1] + kernel_size[1];
    size_t bottom_pad = required_input_height - input_shape[ndim - 3] - padding[0];
    size_t right_pad = required_input_width - input_shape[ndim - 2] - padding[1];
    size_t top_pad = 0;
    size_t left_pad = 0;

    /*
    TODO: Cleanup this nested looped mess
    NOTE: Only update top and left padding after first iteration
            for preventing underflow in index calculation
    */
    for (size_t batch = 0; batch < output_shape[ndim - 4]; ++batch) {// batch iterator
        top_pad = 0;
        for (size_t out_h = 0; out_h < output_shape[ndim - 3]; ++out_h) {// height iterator
            left_pad = 0;
            for (size_t out_w = 0; out_w < output_shape[ndim - 2]; ++out_w) {// width iterator
                for (size_t out_ch = 0; out_ch < output_shape[ndim - 1]; ++out_ch) {
                    size_t output_index = batch * output_strides[ndim - 4] \
                                            + out_h * output_strides[ndim - 3] \
                                            + out_w * output_strides[ndim - 2] \
                                            + out_ch;
    
                    size_t input_index = batch * input_strides[ndim - 4] \
                                            + ((out_h * stride[0]) - top_pad) * input_strides[ndim - 3] \
                                            + ((out_w * stride[1]) - left_pad) * input_strides[ndim - 2] \
                                            + out_ch;

                    T pooled_out = input[input_index];
                    size_t row_offset = input_index;
                    for (size_t k_h = 0; k_h < kernel_size[0]; ++k_h) {
                        if (padding[0]) {
                            if (out_h == 0 && k_h < padding[0]) {
                                continue;
                            } else if (out_h == (output_shape[ndim - 3] - 1)) {
                                if (k_h >= (kernel_size[0] - bottom_pad)) {
                                    break;
                                }
                            }
                        }
                        size_t col_offset = 0;
                        for (size_t k_w = 0; k_w < kernel_size[1]; ++k_w) {
                            if (padding[1]) {
                                if (out_w == 0 && k_w < padding[1]) {
                                    continue;
                                } else if (out_w == (output_shape[ndim - 2] - 1)) {
                                    if (k_w >= (kernel_size[1] - right_pad)) {
                                        break;
                                    }
                                }
                            }
                            
                            size_t comp_index = row_offset + col_offset;
                            if (input[comp_index] > pooled_out) {
                                pooled_out = input[comp_index];
                            }
                            col_offset += input_strides[ndim - 2];
                        }
                        row_offset += input_strides[ndim - 3];
                    }
                    output[output_index] = pooled_out;
                }
                left_pad = padding[1]; 
            }
            top_pad = padding[0];
        }
    }
}
