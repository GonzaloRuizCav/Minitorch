@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.
    Given input tensor of
       `batch, in_channels, width`
    and weight tensor
       `out_channels, in_channels, k_width`
    Computes padded output of
       `batch, out_channels, width`
    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)
    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = weight_strides
    s2 = input_strides

    input_index = np.empty(MAX_DIMS, np.int32)
    out_index = np.empty(MAX_DIMS, np.int32)
    weight_index = np.empty(MAX_DIMS, np.int32)

    Weight_values = in_channels_ * kw

    if reverse is False:
        for i in range(out_size):
            # out_index has dimension (batch, out_channel, width)
            to_index(i, out_shape, out_index)
            res = 0.0
            for weight_pos in range(Weight_values):
                to_index(weight_pos, weight_shape, weight_index)
                to_index(weight_pos, weight_shape, input_index)
                input_index[0] = out_index[0]
                weight_index[0] = out_index[1]
                input_index[2] += out_index[2]
                if input_index[2] < width:
                    Weight_Pos = index_to_position(weight_index, s1)
                    Input_Pos = index_to_position(input_index, s2)
                    res += weight[Weight_Pos] * input[Input_Pos]
            out[i] = res
    else:
        for i in range(out_size):
            # out_index has dimension (batch, out_channel, width)
            to_index(i, out_shape, out_index)
            res = 0.0
            # Just take the values for each convolution: kw * in_channels
            for weight_pos in range(Weight_values):
                # Get both indexes: the index in the input and in the weights
                to_index(weight_pos, weight_shape, weight_index)
                to_index(weight_pos, weight_shape, input_index)
                # Change indexes: put the input index to the batch, and weight index to the out_channels
                input_index[0] = out_index[0]
                weight_index[0] = out_index[1]
                input_index[2] = out_index[2] - weight_index[2]
                Weight_Pos = index_to_position(weight_index, s1)
                Input_Pos = index_to_position(input_index, s2)
                # print("Position in weights", Weight_Pos)
                # print("Position in input", Input_Pos)
                if input_index[2] >= 0:
                    res += weight[Weight_Pos] * input[Input_Pos]
            out[i] = res
