# This visualization shows the three-phase process of blockReduce():

## Phase 1 
(Intra-warp reduction): Each of the 8 warps independently performs warpReduce() on its 32 threads. After this step, only the first thread of each warp (threads 0, 32, 64, 96, 128, 160, 192, 224) holds a meaningful partial sum.

## Phase 2 
(Inter-warp communication): The condition if (lane == 0) ensures only warp leaders write to shared memory. Since lane = threadIdx.x % 32, this means threads 0, 32, 64, etc. write their partial sums to shared[0], shared[1], shared[2], etc. The __syncthreads() barrier ensures all writes complete before proceeding.

## Phase 3 

(Final reduction): The condition threadIdx.x < blockDim.x / WARP_SIZE (which equals 8 for a 256-thread block) means only threads 0-7 participate. These threads load values from shared memory using shared[lane] where lane is their thread ID. Since only warp 0 participates (if (wid == 0)), warpReduce() operates on threads 0-7 with the 8 partial sums.

The algorithm efficiently combines register-based warp reductions with shared memory for inter-warp communication, achieving O(log₂ 256) complexity while maximizing hardware utilization.
