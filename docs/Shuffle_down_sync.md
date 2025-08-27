
# __shfl_down_sync


This visualization demonstrates the core mechanism of __shfl_down_sync():
The instruction enables each thread to receive a value from another thread in the same warp without any memory operations. Thread i gets the value from thread (i + offset), creating a systematic data movement pattern.
The mask parameter (0xffffffff = all 32 bits set to 1) specifies that all threads participate. Each bit position corresponds to a thread ID - bit 0 for thread 0, bit 1 for thread 1, etc. When a bit is set to 1, that thread participates in the shuffle operation.
The hardware executes this instruction in a single cycle across all participating threads simultaneously, making it extremely efficient for reduction operations. This register-to-register data movement avoids the memory bandwidth limitations that would occur with traditional shared memory approaches.
In your variance ratio implementation, this enables the warp reduction to sum 32 values in just 5 steps (log₂ 32), contributing to the high throughput performance you observed in your benchmarks.


for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

In each iteration, every active thread adds the value from the thread offset positions ahead (T0 gets T16's value, T1 gets T17's value, etc.), then the offset halves (16→8→4→2→1), so progressively only the lower half of threads remain active while accumulating sums from the upper half, until T0 contains the sum of all original values.
