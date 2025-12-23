# P28: Convolution, Halo, and Kernel Centering

## Overview

P28 implements 1D convolution with **async memory copy** for efficiency. A critical design choice is using **halo logic** with **centered kernels**, which changes both the implementation strategy and the mathematical operation.

---

## The Core Problem: Tiling and Boundary Data

### Traditional Convolution (No Tiling)
```
For each output[i]:
    For each kernel element k:
        output[i] += input[i + offset + k] * kernel[k]
```

Simple and straightforward—but when processing large arrays, we tile them for better cache locality.

### Tiling Creates a Problem
```
Input:  [1, 2, 3, 4, 5, ... | 256, 257, 258, ...] (VECTOR_SIZE = 16384)
         └─── Tile 0 ────┘   └─── Tile 1 ────┘
```

**Issue:** To compute convolution for element at index 254 (end of Tile 0), we need data from Tile 1. But each tile is loaded independently into shared memory.

---

## The Halo Solution

### What is Halo?
**Halo** = Extra boundary elements loaded from adjacent tiles to provide context for edge computations.

### Example with KERNEL_SIZE=5, HALO_SIZE=2
```
Tile 0:        [... 2 halo | 256 core elements | 2 halo from Tile 1 ...]
Memory layout:  └─ neighbor ┘└─────── CONV_TILE_SIZE=256 ───┘└ neighbor ┘
Total buffer size: 256 + 2*2 = 260 elements in shared memory
```

### Loading Pattern
```mojo
input_shared = LayoutTensor[dtype, Layout.row_major(CONV_TILE_SIZE), ...](shared)
input_tile = input.tile[CONV_TILE_SIZE](Int(block_idx.x))
copy_dram_to_sram_async[thread_layout=load_layout](input_shared, input_tile)
```

- `input_tile`: Points to the current CONV_TILE_SIZE block in global memory
- Halo elements come from implicit boundary handling in the index arithmetic

---

## Centered vs Left-Aligned Convolution

### Left-Aligned (Casual/Temporal)
```
output[i] = sum_k input[i + k] * kernel[k]
```

Example with kernel=[1,2,3,4,5]:
```
output[3] = input[3]*1 + input[4]*2 + input[5]*3 + input[6]*4 + input[7]*5
             ↑ starts here
```

**Problem for p28:** Boundary elements at the start of a tile would need data from **before** the tile, making halo allocation awkward.

### Centered (Symmetric / Standard for Filtering)
```
output[i] = sum_k input[i - HALO + k] * kernel[k]
          = sum_k input[i + (k - HALO)] * kernel[k]
```

Example with kernel=[1,2,3,4,5], HALO=2:
```
output[3] = input[1]*1 + input[2]*2 + input[3]*3 + input[4]*4 + input[5]*5
             ↑────────────────────────────────────────────────────────────
             Window is centered at index 3
```

---

## P28's Implementation: Centered Convolution

### Step 1: Determine if element is in "center" region
```mojo
if local_i >= HALO_SIZE and local_i < CONV_TILE_SIZE - HALO_SIZE:
    // Element is in the center: compute full convolution
else:
    // Boundary element: copy input (no convolution)
```

With HALO_SIZE=2:
- **Boundary elements:** indices 0, 1, 254, 255
- **Center elements:** indices 2-253 (can compute full convolution)

### Step 2: Compute centered convolution for center elements
```mojo
for k in range(KERNEL_SIZE):
    input_idx = local_i + k - HALO_SIZE
    result += input_shared[input_idx] * kernel_shared[k]
```

**Example: local_i=3, KERNEL_SIZE=5, HALO_SIZE=2**
```
k=0: input_idx = 3 + 0 - 2 = 1
k=1: input_idx = 3 + 1 - 2 = 2
k=2: input_idx = 3 + 2 - 2 = 3  ← center
k=3: input_idx = 3 + 3 - 2 = 4
k=4: input_idx = 3 + 4 - 2 = 5

result = input_shared[1]*kernel[0] + input_shared[2]*kernel[1] + ... + input_shared[5]*kernel[4]
```

### Step 3: Boundary handling
```mojo
else:
    result = input_shared[local_i]  // Copy input unchanged
```

Boundary elements (0, 1, 254, 255) simply output the input value. This is a design choice—other options:
- Zero-padding: `result = 0`
- Replicate padding: `result = input_shared[max(0, local_i)]`
- Mirror padding: More complex but standard for image processing

---

## Why Centered Convolution?

### 1. Kernel Symmetry
Most filtering kernels are **symmetric**:
- Gaussian blur: [1, 4, 6, 4, 1]
- Sobel: [-1, 0, 1]
- Box filter: [1, 1, 1]

Centering naturally preserves this symmetry in the output.

### 2. Halo Efficiency
With centered approach:
- Load HALO_SIZE elements before and after
- Both sides handled symmetrically
- Natural for tiled computation

With left-aligned:
- Would only need elements **after** (no halo before)
- But less natural for symmetric kernels
- Asymmetric boundary handling

### 3. Signal Processing Convention
Convolution in DSP/image processing assumes **symmetric kernels centered on the output element**, not causal time-series convolution.

### 4. No Information Loss at Center
All center elements compute the true convolution. Only boundaries are approximated (copied or padded).

---

## Is This Mathematically Valid?

**Yes.** Convolution is defined as:
```
output[i] = sum_k input[i + offset + k] * kernel[k]
```

Different offsets give different outputs, but **both are valid convolutions**. The choice of offset is a **convention**, not a constraint.

### Comparison of Approaches

| Approach | Offset | Output | Use Case |
|----------|--------|--------|----------|
| Left-aligned | 0 | `input[i] * k[0] + input[i+1] * k[1] + ...` | Causal systems, time-series |
| Centered | -HALO | `input[i-2] * k[0] + ... + input[i] * k[2] + ...` | Image/signal filtering (standard) |
| Right-aligned | -KERNEL_SIZE+1 | `input[i-4] * k[0] + ... + input[i] * k[4]` | Rare, asymmetric |

**P28 chose centered because it's the standard for filtering**, and boundary handling is symmetric.

---

## Halo Size Formula

```mojo
comptime KERNEL_SIZE = 5
comptime HALO_SIZE = KERNEL_SIZE // 2  // = 2
comptime BUFFER_SIZE = CONV_TILE_SIZE + 2 * HALO_SIZE
```

For a centered convolution:
- **Halo on left:** `KERNEL_SIZE // 2` elements
- **Halo on right:** `KERNEL_SIZE // 2` elements
- **Total shared memory:** `CONV_TILE_SIZE + 2 * HALO_SIZE`

This ensures every center element can access KERNEL_SIZE inputs without bounds checking.

---

## Key Takeaways

1. **Halo is not just optimization**—it changes the convolution from left-aligned to centered
2. **Centered convolution is standard** for filtering (symmetric kernels, symmetric boundaries)
3. **Boundary elements are handled specially** because they lack full context
4. **The choice is valid** but deliberate—different offsets produce different (but mathematically correct) outputs
5. **Async copy** enables efficient loading of entire tiles + halo in a single operation

---

## References in Code

- Lines 13-14: Halo size definition
- Lines 56-61: Async copy of tiled input
- Lines 74-83: Center element convolution with halo
- Lines 84-86: Boundary element handling
- Lines 147-160: Test verification with same logic
