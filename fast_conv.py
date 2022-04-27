from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


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
    for i in prange(batch_):
        for j in prange(out_channels):
            for k in prange(out_width):
                res = 0.0
                # Out_index is now (i,j,k)
                for xweight in prange(in_channels_):
                    for yweight in prange(kw):
                        # Index in weights is now (j, xweight, yweight)
                        # Index in input is now (i, xweight, Position)
                        Weight_Pos = j * s1[0] + xweight * s1[1] + yweight * s1[2]
                        if reverse is False:
                            Position = yweight + k
                            if Position < width:
                                Input_Pos = (
                                    i * s2[0] + xweight * s2[1] + Position * s2[2]
                                )
                                res += weight[Weight_Pos] * input[Input_Pos]
                        if reverse is True:
                            Position = k - yweight
                            if Position >= 0:
                                Input_Pos = (
                                    i * s2[0] + xweight * s2[1] + Position * s2[2]
                                )
                                res += weight[Weight_Pos] * input[Input_Pos]
                Out_Pos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]
                out[Out_Pos] = res


class Conv1dFun(Function):
    """
    Compute a 1D Convolution.
    Args:
        ctx: Context.
        input (:class:'Tensor'): batch x in_channel x h x w.
        weight (:class:'Tensor'): out_channel x in_channel x kh x kw.
    Returns:
        (:class:'Tensor'): batch x out_channel x h x w.
    """

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True, fastmath=True)
def tensor_conv2d(
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
    2D Convolution implementation.
    Given input tensor of
       `batch, in_channels, height, width`
    and weight tensor
       `out_channels, in_channels, k_height, k_width`
    Computes padded output of
       `batch, out_channels, height, width`
    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
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
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    for i in prange(batch_):
        for j in prange(out_channels_):
            for k in prange(out_shape[2]):
                for l in prange(out_shape[3]):
                    res = 0.0
                    Out_Pos = (
                        i * out_strides[0]
                        + j * out_strides[1]
                        + k * out_strides[2]
                        + l * out_strides[3]
                    )
                    for zweight in range(in_channels_):
                        for xweight in range(kh):
                            for yweight in range(kw):
                                Weight_Pos = (
                                    j * s20
                                    + zweight * s21
                                    + xweight * s22
                                    + yweight * s23
                                )
                                if not reverse:
                                    if xweight + k < height and yweight + l < width:
                                        Input_Pos = (
                                            i * s10
                                            + zweight * s11
                                            + (xweight + k) * s12
                                            + (yweight + l) * s13
                                        )
                                        res += weight[Weight_Pos] * input[Input_Pos]
                                else:
                                    if k - xweight >= 0 and l - yweight >= 0:
                                        Input_Pos = (
                                            i * s10
                                            + zweight * s11
                                            + (k - xweight) * s12
                                            + (l - yweight) * s13
                                        )
                                        res += weight[Weight_Pos] * input[Input_Pos]
                    out[Out_Pos] = res


class Conv2dFun(Function):
    """
    Compute a 1D Convolution.
    Args:
        ctx: Context.
        input (:class:'Tensor'): batch x in_channel x h x w.
        weight (:class:'Tensor'): out_channel x in_channel x kh x kw.
    Returns:
        (:class:'Tensor'): batch x out_channel x h x w.
    """

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
