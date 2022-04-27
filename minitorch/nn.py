from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    nh = height // kh
    nw = width // kw
    inp = (
        input.contiguous()
        .view(batch, channel, nh, kh, nw, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    i = inp.view(batch, channel, nh, nw, kh * kw)
    return (
        i,
        height // kh,
        width // kw,
    )


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    kh, kw = kernel
    batch, channel, height, width = input.shape
    t, nh, nw = tile(input, (kh, 1))
    tensor2 = t.mean(4).view(batch, channel, nh, nw)
    t2, nh2, nw2 = tile(tensor2, (1, kw))
    Out = t2.mean(4).view(batch, channel, nh2, nw2)
    return Out


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim)


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # Process: first, map with exp. Then, reduce it. Then, just divide one against the other
    a = input.exp()
    return a / a.sum(dim)


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    return softmax(input, dim).log()


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    kh, kw = kernel
    batch, channel, height, width = input.shape
    t, nh, nw = tile(input, (kh, 1))
    tensor2 = max(t, 4).view(batch, channel, nh, nw)
    t2, nh2, nw2 = tile(tensor2, (1, kw))
    Out = max(t2, 4).view(batch, channel, nh2, nw2)
    return Out


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore is False:
        return (rand(input.shape, backend=input.backend) > rate) * input
    else:
        return input
