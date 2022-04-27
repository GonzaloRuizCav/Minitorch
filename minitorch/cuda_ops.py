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

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.3.
        x = cuda.blockIdx.x * THREADS_PER_BLOCK + cuda.threadIdx.x
        if x < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(x, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.3.
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if x < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(x, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3
    x = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x
    local_idx = cuda.threadIdx.x
    a_shared = numba.cuda.shared.array(BLOCK_DIM, numba.float64)
    # Padding
    if x < size:
        a_shared[local_idx] = float(a[x])
    else:
        a_shared[local_idx] = 0
    # Implement the sum
    for i in [1, 2, 4, 8, 16]:
        numba.cuda.syncthreads()
        if local_idx % (i * 2) == 0:
            a_shared[local_idx] = a_shared[local_idx] + a_shared[local_idx + i]
    out[cuda.blockIdx.x] = a_shared[0]


jit_sum_practice = cuda.jit()(_sum_practice)

# numba.cuda.local and numba.cuda.shared (usually use numba.cuda.shared)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):

        BLOCK_DIM = 1024
        # TODO: Implement for Task 3.3.
        a_shared = numba.cuda.shared.array(BLOCK_DIM, numba.float64)
        # reduce_size = a_shape[reduce_dim]
        local_idx = cuda.threadIdx.x
        reduce_size = a_shape[reduce_dim]

        # Padding
        if local_idx >= reduce_size:
            a_shared[local_idx] = reduce_value
        cuda.syncthreads()

        if local_idx < reduce_size:
            out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            # Map each block to the shared memory
            to_index(cuda.blockIdx.x, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            out_index[reduce_dim] = local_idx
            a_pos = index_to_position(out_index, a_strides)
            cuda.syncthreads()
            a_shared[local_idx] = a_storage[a_pos]
            cuda.syncthreads()
            # Implement the sum on each block and save the result into the output vector
            i = 1
            while i < reduce_size:
                cuda.syncthreads()
                if local_idx % (i * 2) == 0:
                    a_shared[local_idx] = fn(
                        a_shared[local_idx], a_shared[local_idx + i]
                    )
                i = i * 2
            cuda.syncthreads()
            out[out_pos] = a_shared[0]

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    x = cuda.blockIdx.x * THREADS_PER_BLOCK + cuda.threadIdx.x
    y = cuda.blockIdx.y * THREADS_PER_BLOCK + cuda.threadIdx.y
    localX = cuda.threadIdx.x
    localY = cuda.threadIdx.y
    sharedA = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sharedB = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    if x < size and y < size:
        sharedA[localX, localY] = a[x * size + y]
        sharedB[localX, localY] = b[x * size + y]
    else:
        sharedA[localX, localY] = 0
        sharedB[localX, localY] = 0
    cuda.syncthreads()

    if x < size:
        if y < size:
            local_i = cuda.threadIdx.x
            local_j = cuda.threadIdx.y
            temp = 0
            for k in range(size):
                temp += sharedA[local_i, k] * sharedB[k, local_j]
            out[x * size + y] = temp


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    # Define all the indexes required for the algorithm
    x = numba.cuda.blockIdx.x * BLOCK_DIM + numba.cuda.threadIdx.x
    y = numba.cuda.blockIdx.y * BLOCK_DIM + numba.cuda.threadIdx.y
    z = numba.cuda.blockIdx.z + numba.cuda.threadIdx.z

    localX = numba.cuda.threadIdx.x
    localY = numba.cuda.threadIdx.y
    # Create both shared memories with the block size
    a_shared = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float64)
    b_shared = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=numba.float64)
    if x >= out_shape[1] or y >= out_shape[2]:
        a_shared[localX, localY] = 0
        b_shared[localX, localY] = 0
    # k will be the starting position of each block in a and b
    temp = 0.0
    for k in range(0, a_shape[2], BLOCK_DIM):
        cuda.syncthreads()
        # Filling a_shared and b_shared only when the positions are in bounds.
        a_pos = z * a_batch_stride + x * a_strides[1] + (k + localY) * a_strides[2]
        b_pos = z * b_batch_stride + y * b_strides[2] + (k + localX) * b_strides[1]
        if k + localY < a_shape[2]:
            a_shared[localX, localY] = a_storage[a_pos]
        if k + localX < b_shape[1]:
            b_shared[localX, localY] = b_storage[b_pos]

        cuda.syncthreads()

        # Perform the operation of matrix multiply
        for i in range(0, BLOCK_DIM):
            if k + i < a_shape[2]:
                temp += a_shared[localX, i] * b_shared[i, localY]
    # Copy the values to the output only when the threads are active.
    if x < out_shape[1] and y < out_shape[2]:
        out_pos = z * out_strides[0] + x * out_strides[1] + y * out_strides[2]
        out[out_pos] = temp


def matrix_multiply(a, b):
    """
    Tensor matrix multiply
    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::
        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        out.shape[0],
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
