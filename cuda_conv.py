from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from .tensor_functions import Function

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


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
    1D Cuda Convolution implementation.
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

    # Block arrangement is going to be size of output: (size_out, )
    globalPos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    globalIndex = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(globalPos, out_shape, globalIndex)
    globalX = globalIndex[0]
    globalY = globalIndex[1]
    globalZ = globalIndex[2]
    # Im going to have kw * in_channels threads per block: [in_channels, kw]
    # Initialize the shared memories
    Shared_Input = cuda.local.array((in_channels_, kw), numba.int32)
    Shared_Weights = cuda.local.array((in_channels_, kw), numba.int32)
    # And get the local indexes of the threads
    localX = cuda.threadIdx.x
    localY = cuda.threadIdx.y

    # Time to fill the memories. The weight one is straightforward
    Shared_Weights[localX, localY] = weight[
        globalY * s1[0] + localX * s1[1] + localY * s1[2]
    ]
    # The input memory is not as straightforward
    if reverse is False:
        Pos = globalZ + localY
        if Pos < width:
            Shared_Input[localX, localY] = input[
                globalX * s2[0] + localX * s2[1] + Pos * s2[2]
            ]
        else:
            Shared_Input[localX, localY] = 0
        # Once the shared memories are initialized we just compute the sum and accumulate
        Res = 0.0
        # Wait for all threads to reach this point
        numba.cuda.syncthreads()
        Res += Shared_Input[localX, localY] * Shared_Weights[localX, localY]
    else:
        Pos = globalZ - localY
        if Pos >= 0:
            Shared_Input[localX, localY] = input[
                globalX * s2[0] + localX * s2[1] + Pos * s2[2]
            ]
        else:
            Shared_Input[localX, localY] = 0
        Res = 0.0
        # Wait for all threads to reach this point
        numba.cuda.syncthreads()
        Res += Shared_Input[localX, localY] * Shared_Weights[localX, localY]


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


conv1d = cuda.jit()(Conv1dFun)


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

    # Block arrangement is going to be size of output: (size_out, )
    globalPos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    globalIndex = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(globalPos, out_shape, globalIndex)
    globalX = globalIndex[0]
    globalY = globalIndex[1]
    globalZ = globalIndex[2]
    globalK = globalIndex[3]
    # Initialize the shared memories
    Shared_Input = cuda.local.array((in_channels_, kh, kw), numba.int32)
    Shared_Weights = cuda.local.array((in_channels_, kh, kw), numba.int32)
    # And get the local indexes of the threads
    localX = cuda.threadIdx.x
    localY = cuda.threadIdx.y
    localZ = cuda.threadIdx.z

    # Input index is: globalY, localZ, Pos1, Pos2
    # Weight index is: globalY, localZ, localX, localY

    # Time to fill the memories. The weight one is straightforward
    Shared_Weights[localX, localY, localZ] = weight[
        globalY * s10 + localZ * s11 + localX * s12 + localY * s13
    ]
    # The input memory is not as straightforward
    if reverse is False:
        Pos1 = globalZ + localX
        Pos2 = globalK + localY
        if Pos1 < height and Pos2 < width:
            Shared_Input[localX, localY, localZ] = input[
                globalX * s20 + localZ * s21 + Pos1 * s22 + Pos2 * s23
            ]
        else:
            Shared_Input[localX, localY, localZ] = 0
        # Once the shared memories are initialized we just compute the sum and accumulate
        Res = 0.0
        # Wait for all threads to reach this point
        numba.cuda.syncthreads()
        Res += Shared_Input[localX, localY] * Shared_Weights[localX, localY]
    else:
        Pos1 = globalZ - localX
        Pos2 = globalK - localY
        if Pos1 >= 0 and Pos2 >= 0:
            Shared_Input[localX, localY, localZ] = input[
                globalX * s20 + localZ * s21 + Pos1 * s22 + Pos2 * s23
            ]
        else:
            Shared_Input[localX, localY, localZ] = 0
        Res = 0.0
        # Wait for all threads to reach this point
        numba.cuda.syncthreads()
        Res += Shared_Input[localX, localY] * Shared_Weights[localX, localY]


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


conv2d = cuda.jit()(Conv2dFun)
