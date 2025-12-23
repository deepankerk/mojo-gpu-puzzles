from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext
from gpu.host.compile import get_gpu_target
from layout import Layout, LayoutTensor
from utils import IndexList
from math import log2
from algorithm.functional import elementwise, vectorize
from sys import simd_width_of, argv, align_of
from testing import assert_equal
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: elementwise_add
comptime SIZE = 1024
comptime rank = 1
comptime layout = Layout.row_major(SIZE)
comptime dtype = DType.float32
comptime SIMD_WIDTH = simd_width_of[dtype, target = get_gpu_target()]()


fn elementwise_add[
    layout: Layout, dtype: DType, simd_width: Int, rank: Int, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn add[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        #     GPU Architecture:
        # ├── Grid (entire problem)
        # │   ├── Block 1 (multiple warps)
        # │   │   ├── Warp 1 (32 threads) --> We'll learn about Warp in the next Part VI
        # │   │   │   ├── Thread 1 → SIMD[4 elements]  ← Our focus (GPU-dependent width)
        # │   │   │   ├── Thread 2 → SIMD[4 elements]
        # │   │   │   └── ...
        # │   │   └── Warp 2 (32 threads)
        # │   └── Block 2 (multiple warps)
        
        # Above is the key aspect where we are essentially saying each thread will process 4 elements
        # We could used the standard approach of using thread/block indexing but we are moving to a higher level here
        # Naive Implementation 1
        # for i in range(simd_width):
        #     idx = indices[0] + i
        #     if idx < size:
        #         output[idx] = a[idx] + b[idx]
        
        # Implementation 2, Vectorized
        idx = indices[0]                                  # Linear index: 0, 4, 8, 12... (GPU-dependent spacing)
        a_simd = a.aligned_load[simd_width](idx, 0)       # Load: [a[0:4], a[4:8], a[8:12]...] (4 elements per load)
        b_simd = b.aligned_load[simd_width](idx, 0)       # Load: [b[0:4], b[4:8], b[8:12]...] (4 elements per load)
        ret = a_simd + b_simd                             # SIMD: 4 additions in parallel (GPU-dependent)
        output.aligned_store[simd_width](idx, 0, ret)     # Store: 4 results simultaneously (GPU-dependent)

        #  Traditional GPU:
        # ├─ ALU 0: adds a[0] + b[0]
        # ├─ ALU 1: adds a[1] + b[1]
        # ├─ ALU 2: adds a[2] + b[2]
        # └─ ALU 3: adds a[3] + b[3]
        # (4 cores processing in parallel = 4 threads)

        # SIMD-aware GPU:
        # ├─ Vector ALU (256-bit): adds [a[0:4]] + [b[0:4]] simultaneously
        # └─ Single vector register holds all 4 results
        # (1 thread, 1 vector instruction, 4x throughput)
        # Traditional CUDA (hand-written):
        # int idx = blockIdx.x * blockDim.x + threadIdx.x;
        # output[idx] = a[idx] + b[idx];  // Scalar code
        # Compiler sees 32 scalar operations → generates:

        # ld.f32   %f0, [a[idx]]          // load scalar
        # ld.f32   %f1, [b[idx]]          // load scalar
        # add.f32  %f2, %f0, %f1          // add scalar
        # st.f32   [output[idx]], %f2     // store scalar
        # (Repeated 32 times per warp)

        # Mojo explicit SIMD:

        # a_simd = a.aligned_load[4](idx, 0)  // Vector code
        # ret = a_simd + b_simd
        # output.aligned_store[4](idx, 0, ret)
        # Compiler sees vector operations → generates:

        # ld.v4.f32  %v0, [a[idx]]        // load 4-element vector (single instruction)
        # ld.v4.f32  %v1, [b[idx]]        // load 4-element vector
        # add.v4.f32 %v2, %v0, %v1        // add 4 elements (single instruction)
        # st.v4.f32  [output[idx]], %v2   // store 4-element vector
        # Both compile to CUDA/PTX, but:

        # Traditional: Compiler must recognize pattern and auto-vectorize (often fails)
        # Mojo explicit: Compiler sees vector ops directly → guaranteed vector instructions
        # The real advantage: Mojo's explicit syntax forces the compiler to generate vector code, instead of hoping it auto-vectorizes. The backend is the same, but the generated instructions are better.

    elementwise[add, SIMD_WIDTH, target="gpu"](size, ctx)


# ANCHOR_END: elementwise_add


# ANCHOR: tiled_elementwise_add
comptime TILE_SIZE = 32


fn tiled_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn process_tiles[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        output_tile = output.tile[tile_size](tile_id)
        # Each tile[size](id) creates a view into the original tensor
        # Views are zero-copy - no data movement, just pointer arithmetic
        a_tile = a.tile[tile_size](tile_id)
        b_tile = b.tile[tile_size](tile_id)
        # Naive Implementation 1
        # ret = a_tile + b_tile
        # for i in range(tile_size):
        #     output_tile[i] = ret[i]

        # Implementation 2
        # Cache optimization: Consecutive memory accesses maximize cache hit rates
        # Compiler optimization: @parameter loops unroll completely at compile-time
            # @parameter means the loop bounds are compile-time constants, so the compiler can completely unroll the loop.
        # Memory bandwidth: Sequential access aligns with memory controller design
        # Performance trade-offs:

        # Fewer logical threads: May not fully utilize all GPU cores at low occupancy
        # More work per thread: Better cache utilization and reduced coordination overhead
        # Sequential access: Optimal memory bandwidth utilization within each thread
        # Reduced overhead: Less thread launch and coordination overhead
        # Important note: “Fewer threads” refers to the logical programming model. The GPU scheduler can still achieve high hardware utilization by running multiple warps and efficiently switching between them during memory stalls.
        # https://puzzles.modular.com/puzzle_23/tile.html#6-performance-characteristics (This part is important to understand because of how trade offs are)
        @parameter
        for i in range(tile_size):
            a_vec = a_tile.load[simd_width](i, 0)
            b_vec = b_tile.load[simd_width](i, 0)
            ret = a_vec + b_vec
            output_tile.store[simd_width](i, 0, ret)

    num_tiles = (size + tile_size - 1) // tile_size
    elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)


# ANCHOR_END: tiled_elementwise_add


# ANCHOR: manual_vectorized_tiled_elementwise_add
fn manual_vectorized_tiled_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size groups of simd_width elements
    comptime chunk_size = tile_size * simd_width

    @parameter
    @always_inline
    fn process_manual_vectorized_tiles[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        output_tile = output.tile[chunk_size](tile_id)
        a_tile = a.tile[chunk_size](tile_id)
        b_tile = b.tile[chunk_size](tile_id)

        # elemenwise_add
        # idx = indices[0]                                  # Linear index: 0, 4, 8, 12... (GPU-dependent spacing)
        # a_simd = a.aligned_load[simd_width](idx, 0)       # Load: [a[0:4], a[4:8], a[8:12]...] (4 elements per load)
        # b_simd = b.aligned_load[simd_width](idx, 0)       # Load: [b[0:4], b[4:8], b[8:12]...] (4 elements per load)
        # ret = a_simd + b_simd                             # SIMD: 4 additions in parallel (GPU-dependent)
        # output.aligned_store[simd_width](idx, 0, ret)     # Store: 4 results simultaneously (GPU-dependent)

        @parameter
        for i in range(tile_size):
            # aligned_load[simd_width](index, offset) loads multiple elements as a vector from a tensor.
            # Parameters:
            # simd_width: How many elements to load (4, 8, 16, etc.)
            # index: Starting position in the tensor
            # offset: Additional offset (usually 0)
            global_start = tile_id * chunk_size + i * simd_width

            a_vec = a.aligned_load[simd_width](global_start, 0)
            b_vec = b.aligned_load[simd_width](global_start, 0)
            ret = a_vec + b_vec

            output.aligned_store[simd_width](global_start, 0, ret)

    # Number of tiles needed: each tile processes chunk_size elements
    num_tiles = (size + chunk_size - 1) // chunk_size
    elementwise[
        process_manual_vectorized_tiles, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: manual_vectorized_tiled_elementwise_add


# ANCHOR: vectorize_within_tiles_elementwise_add
fn vectorize_within_tiles_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size elements (not SIMD groups)
    @parameter
    @always_inline
    fn process_tile_with_vectorize[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        tile_start = tile_id * tile_size
        tile_end = min(tile_start + tile_size, size)
        actual_tile_size = tile_end - tile_start

        fn vectorized_add[
            width: Int
        ](i: Int) unified {read tile_start, read a, read b, mut output}:
            global_idx = tile_start + i
            if global_idx + width <= size:
                a_vec = a.aligned_load[width](global_idx, 0)
                b_vec = b.aligned_load[width](global_idx, 0)
                result = a_vec + b_vec
                output.aligned_store[width](global_idx, 0, result)

        # Use vectorize within each tile
        # Essentially, the function runs actual tile size times with SIMD width.
        # This approach is not very different from the tiled approach that we have covered, just with the use of vectorized function.  
        vectorize[simd_width](actual_tile_size, vectorized_add)



    num_tiles = (size + tile_size - 1) // tile_size
    elementwise[
        process_tile_with_vectorize, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: vectorize_within_tiles_elementwise_add


@parameter
@always_inline
fn benchmark_elementwise_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    bench_ctx = DeviceContext()
    comptime layout = Layout.row_major(test_size)
    out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1

    a_tensor = LayoutTensor[mut=False, dtype, layout, MutAnyOrigin](
        a.unsafe_ptr()
    )
    b_tensor = LayoutTensor[mut=False, dtype, layout, MutAnyOrigin](
        b_buf.unsafe_ptr()
    )
    out_tensor = LayoutTensor[mut=True, dtype, layout, MutAnyOrigin](
        out.unsafe_ptr()
    )

    @parameter
    @always_inline
    fn elementwise_workflow(ctx: DeviceContext) raises:
        elementwise_add[layout, dtype, SIMD_WIDTH, rank, test_size](
            out_tensor, a_tensor, b_tensor, ctx
        )

    b.iter_custom[elementwise_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
fn benchmark_tiled_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    bench_ctx = DeviceContext()
    comptime layout = Layout.row_major(test_size)
    out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1

    a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
    out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

    @parameter
    @always_inline
    fn tiled_workflow(ctx: DeviceContext) raises:
        tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[tiled_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
fn benchmark_manual_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    bench_ctx = DeviceContext()
    comptime layout = Layout.row_major(test_size)
    out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1

    a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
    out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

    @parameter
    @always_inline
    fn manual_vectorized_workflow(ctx: DeviceContext) raises:
        manual_vectorized_tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[manual_vectorized_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
fn benchmark_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    bench_ctx = DeviceContext()
    comptime layout = Layout.row_major(test_size)
    out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1

    a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
    out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

    @parameter
    @always_inline
    fn vectorized_workflow(ctx: DeviceContext) raises:
        vectorize_within_tiles_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[vectorized_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


def main():
    ctx = DeviceContext()
    out = ctx.enqueue_create_buffer[dtype](SIZE)
    out.enqueue_fill(0)
    a = ctx.enqueue_create_buffer[dtype](SIZE)
    a.enqueue_fill(0)
    b = ctx.enqueue_create_buffer[dtype](SIZE)
    b.enqueue_fill(0)
    expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
    expected.enqueue_fill(0)

    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i in range(SIZE):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1
            expected[i] = a_host[i] + b_host[i]

    a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, layout](b.unsafe_ptr())

    ctx.synchronize()

    print("SIZE:", SIZE)
    print("simd_width:", SIMD_WIDTH)

    if argv()[1] == "--elementwise":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        elementwise_add[layout, dtype, SIMD_WIDTH, rank, SIZE](
            out_tensor, a_tensor, b_tensor, ctx
        )
        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--tiled":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        tiled_elementwise_add[layout, dtype, SIMD_WIDTH, rank, SIZE, TILE_SIZE](
            out_tensor, a_tensor, b_tensor, ctx
        )

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--manual-vectorized":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        manual_vectorized_tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--vectorized":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        vectorize_within_tiles_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--benchmark":
        print("Running P21 GPU Benchmarks...")
        print("SIMD width:", SIMD_WIDTH)
        print("-" * 80)
        bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
        bench = Bench(bench_config.copy())

        print("Testing SIZE=16, TILE=4")
        bench.bench_function[benchmark_elementwise_parameterized[16, 4]](
            BenchId("elementwise_16_4")
        )
        bench.bench_function[benchmark_tiled_parameterized[16, 4]](
            BenchId("tiled_16_4")
        )
        bench.bench_function[benchmark_manual_vectorized_parameterized[16, 4]](
            BenchId("manual_vectorized_16_4")
        )
        bench.bench_function[benchmark_vectorized_parameterized[16, 4]](
            BenchId("vectorized_16_4")
        )

        print("-" * 80)
        print("Testing SIZE=128, TILE=16")
        bench.bench_function[benchmark_elementwise_parameterized[128, 16]](
            BenchId("elementwise_128_16")
        )
        bench.bench_function[benchmark_tiled_parameterized[128, 16]](
            BenchId("tiled_128_16")
        )
        bench.bench_function[
            benchmark_manual_vectorized_parameterized[128, 16]
        ](BenchId("manual_vectorized_128_16"))

        print("-" * 80)
        print("Testing SIZE=128, TILE=16, Vectorize within tiles")
        bench.bench_function[benchmark_vectorized_parameterized[128, 16]](
            BenchId("vectorized_128_16")
        )

        print("-" * 80)
        print("Testing SIZE=1048576 (1M), TILE=1024")
        bench.bench_function[
            benchmark_elementwise_parameterized[1048576, 1024]
        ](BenchId("elementwise_1M_1024"))
        bench.bench_function[benchmark_tiled_parameterized[1048576, 1024]](
            BenchId("tiled_1M_1024")
        )
        bench.bench_function[
            benchmark_manual_vectorized_parameterized[1048576, 1024]
        ](BenchId("manual_vectorized_1M_1024"))
        bench.bench_function[benchmark_vectorized_parameterized[1048576, 1024]](
            BenchId("vectorized_1M_1024")
        )

        print(bench)
        print("Benchmarks completed!")

    else:
        print(
            "Usage: --elementwise | --tiled | --manual-vectorized |"
            " --vectorized | --benchmark"
        )
