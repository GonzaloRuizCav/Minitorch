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
            for weight_pos in range(Weight_values):
                to_index(weight_pos, weight_shape, weight_index)
                to_index(weight_pos, weight_shape, input_index)
                input_index[0] = out_index[0]
                weight_index[0] = out_index[1]
                input_index[2] = out_index[2] - weight_index[2]
                Weight_Pos = index_to_position(weight_index, s1)
                Input_Pos = index_to_position(input_index, s2)
                if input_index[2] >= 0:
                    res += weight[Weight_Pos] * input[Input_Pos]
            out[i] = res