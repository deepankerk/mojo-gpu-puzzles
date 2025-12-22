from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = Layout.row_major(SIZE)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # Implementation 1 (My original implementation)
    # The bug I had was related to how I used the barrier in each loop for reduce operations
    local_i = thread_idx.x
    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var val: Scalar[dtype] = min_finite[dtype]()
    if local_i < SIZE:
        shared_max[local_i] = input[local_i]
        val = rebind[Scalar[dtype]](input[local_i])
    else:
        shared_sum[local_i] = 0.0
        shared_max[local_i] = 0.0
    
    # All threads have loaded the data in shared memory
    barrier()

    # all reduce to find the max
    stride = UInt(BLOCK_DIM_X // 2)
    while stride > 0:
        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i + stride], shared_max[local_i])
            # Let threads finish before coalescing further
        barrier()
        stride = stride // 2

    var exp_val: Scalar[dtype] = 0.0
    if local_i < SIZE:
        exp_val = rebind[Scalar[dtype]](exp(val - shared_max[0]))
    shared_sum[local_i] = exp_val
    barrier()
    
    # all reduce to find the sum
    stride = UInt(BLOCK_DIM_X // 2)
    while stride > 0:
        if local_i < stride:
            shared_sum[local_i] += shared_sum[local_i + stride]
            # Let threads finish before coalescing further
        barrier()
        stride = stride // 2

    if local_i < SIZE:
        output[local_i] = exp_val / shared_sum[0]


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    max_val = input[0]
    for i in range(1, SIZE):
        if input[i] > max_val:
            max_val = input[i]
    
    var local_sum: output.element_type = 0
    var exp_input = LayoutTensor[dtype, layout, MutAnyOrigin].stack_allocation()
    for i in range(SIZE):
        exp_input[i] = input[i] - max_val
        exp_input[i] = exp(exp_input[i])
        local_sum += exp_input[i]

    for i in range(SIZE):
        output[i] = exp_input[i] / local_sum

# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
