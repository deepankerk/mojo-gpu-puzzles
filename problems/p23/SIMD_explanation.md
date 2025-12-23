# SIMD vs Thread-Level Parallelism: GPU Programming Models

## Quick Summary

**SIMD (Single Instruction Multiple Data)**: One instruction processes multiple data elements simultaneously using vector hardware.

**Thread-Level Parallelism**: Multiple threads each process one element, coordinated by the GPU scheduler.

Both use SIMD hardware internally, but Mojo explicitly leverages it while traditional CUDA hides it.

---

## Traditional GPU Programming (CUDA)

```cuda
__global__ void add_kernel(float* output, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];  // One element per thread
    }
}
```

**What happens at GPU hardware level:**
- 32 threads in a warp execute the same instruction simultaneously
- Each thread loads/adds/stores its own element
- Hardware coalesces consecutive memory addresses into 1-4 transactions

```
Thread 0: load a[0]    → add → store output[0]
Thread 1: load a[1]    → add → store output[1]
...
Thread 31: load a[31]  → add → store output[31]

Hardware: Detects [a[0], a[1], ..., a[31]] are consecutive
Result: Coalesces 32 individual load instructions → ~1-4 memory transactions
```

**Generated code (PTX assembly):**
```
ld.f32   %f0, [a + 0]     // load scalar
ld.f32   %f1, [b + 0]     // load scalar
add.f32  %f2, %f0, %f1    // add scalar
st.f32   [output + 0], %f2 // store scalar
(repeated 32 times per warp)
```

---

## Mojo Explicit SIMD

```mojo
fn add[simd_width: Int, ...](indices: IndexList[rank]) capturing -> None:
    idx = indices[0]  # Base index for this batch (0, 4, 8, 12...)
    a_simd = a.aligned_load[simd_width](idx, 0)      # Load 4 elements at once
    b_simd = b.aligned_load[simd_width](idx, 0)      # Load 4 elements at once
    ret = a_simd + b_simd                             # Vector add: 4 pairs in parallel
    output.aligned_store[simd_width](idx, 0, ret)    # Store 4 results at once
```

**What happens at GPU hardware level:**
- 1 thread processes 4 elements using vector registers
- Single vector load instruction loads [a[0], a[1], a[2], a[3]] at once
- Single vector ALU executes all 4 additions simultaneously

```
Thread 0: load [a[0], a[1], a[2], a[3]]      (1 vector instruction)
          add [with b[0:4]]                   (1 vector instruction)
          store [to output[0:4]]              (1 vector instruction)
```

**Generated code (PTX assembly):**
```
ld.v4.f32  %v0, [a + 0]        // load 4-element vector (single instruction)
ld.v4.f32  %v1, [b + 0]        // load 4-element vector
add.v4.f32 %v2, %v0, %v1       // add 4 pairs (single instruction)
st.v4.f32  [output + 0], %v2   // store 4-element vector (single instruction)
```

---

## Performance Comparison

| Aspect | CUDA Thread Model | Mojo Explicit SIMD |
|--------|------|------|
| Threads needed | 32 | 1 |
| Load instructions issued | 32 | 1 |
| Add instructions issued | 32 | 1 |
| Store instructions issued | 32 | 1 |
| **Total instructions** | **96** | **3** |
| Memory transactions | ~1-4 (coalesced) | 1 (aligned) |
| Register usage | 32 × per-thread regs | 4 × per-thread regs (more efficient) |
| Instruction cache pressure | High | Low |
| Compiler optimization | Implicit (auto-vectorize) | Explicit (guaranteed) |

**Instruction count advantage**: Mojo reduces from 96 to 3 instructions = **32x fewer instructions to issue**

---

## Why Traditional CUDA Hides SIMD

1. **Abstraction**: Easier to understand "one thread, one element" than vector registers
2. **Portability**: Thread model works across GPU vendors and generations
3. **Auto-vectorization**: Compiler _attempts_ to recognize patterns and generate vector code
4. **Backwards compatibility**: Existing CUDA code remains valid

**Problem with auto-vectorization**: Compilers can't always recognize opportunities, leading to missed optimizations.

---

## Why Mojo Exposes SIMD Explicitly

1. **Performance-critical**: ML workloads need every bit of efficiency
2. **Guaranteed optimization**: Explicit code → compiler generates optimal vector instructions
3. **Modern hardware**: All GPUs now have consistent SIMD support
4. **Control**: Developers don't rely on compiler guessing

---

## GPU Hardware Supporting SIMD

**Vector registers**: Modern GPUs have wide registers (128-bit, 256-bit, 512-bit)
```
Scalar register:  [32-bit value]
Vector register:  [32-bit | 32-bit | 32-bit | 32-bit] = 128-bit total (4 floats)
```

**Vector execution units (VEUs)**: Specialized hardware executes vector instructions
```
Scalar ALU:    a + b → 1 result per cycle
Vector ALU:    [a0, a1, a2, a3] + [b0, b1, b2, b3] → [r0, r1, r2, r3] in 1 cycle
```

**Wide memory buses**: Memory system fetches/stores multiple elements in one transaction
```
Scalar: Load 32-bit → 1 element, ~200 cycles latency
Vector: Load 128-bit → 4 elements, ~200 cycles latency (4x throughput)
```

---

## Key Insight

**Both CUDA and Mojo use SIMD**, but at different abstraction levels:

- **CUDA (implicit)**: You write scalar code for 32 threads → hardware/compiler coalesces them into vector operations
- **Mojo (explicit)**: You write vector code directly → guaranteed vector instructions generated

**Result**: Mojo generates 32x fewer instructions and explicitly controls SIMD, avoiding compiler optimization failures.

---

## Example: Element-wise Addition

**Inefficient (your initial version):**
```mojo
for i in range(simd_width):           # Loop 4 times
    idx = indices[0] + i
    if idx < size:                     # Branch per element
        output[idx] = a[idx] + b[idx]  # 4 separate load/add/store cycles
```

**Efficient (Mojo SIMD):**
```mojo
a_simd = a.aligned_load[simd_width](idx, 0)  # 1 vector load
ret = a_simd + b_simd                        # 1 vector add
output.aligned_store[simd_width](idx, 0, ret) # 1 vector store
```

**Performance**: SIMD version is 3-4x faster due to:
- Reduced instructions (3 vs 12)
- Eliminated branch penalties
- Better memory coalescing
- Higher instruction throughput
