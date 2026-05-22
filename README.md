# Conway's Game of Life — CUDA vs Serial CPU

A side-by-side implementation of Conway's Game of Life that runs the simulation on both the GPU (CUDA) and the CPU (serial), validates that they produce identical results, and reports a speedup measurement.

## What it does

1. Generates a random initial grid of "alive" (black, `0`) and "dead" (white, `255`) cells using OpenCV's `cv::Mat`.
2. Runs the simulation in parallel on the GPU using a CUDA kernel.
3. Runs the same simulation serially on the CPU.
4. Compares the final grids cell-by-cell to verify correctness.
5. Prints GPU time, CPU time, and the resulting speedup.

## Usage

```
./gameoflife <ny> <nx> <maxiter> <showtime> <block_size_x> <block_size_y>
```

| Argument | Meaning |
|---|---|
| `ny`, `nx` | Grid dimensions (max 1024 each, enforced via `assert`) |
| `maxiter` | Number of generations to simulate |
| `showtime` | Display the grid every N iterations (`0` disables display) |
| `block_size_x`, `block_size_y` | CUDA thread-block dimensions |

The grid size is used to derive a 2D CUDA grid: `gridx = ceil(nx / block_size_x)`, `gridy = ceil(ny / block_size_y)`.

## Key components

### `UpdateEquations` (CUDA kernel)
Each thread maps to one cell. It counts the cell's eight neighbors (with wrap-around at the grid edges to make the world toroidal), then applies the standard Life rules:

- **Alive** with `<= 1` or `>= 4` live neighbors → dies (underpopulation / overcrowding)
- **Alive** with 2 or 3 live neighbors → survives
- **Dead** with exactly 3 live neighbors → becomes alive (reproduction)

The kernel reads from `current_state` and writes to `next_state` to avoid race conditions. After each iteration, `cudaMemcpyDeviceToDevice` swaps the buffers for the next pass.

### CPU reference implementation
A straightforward triple-nested loop (`iter` → `iy` → `ix`) using `cv::Mat::at<uchar>()`, applying the same rules. This exists purely as ground truth for validation.

### Timing
- GPU: `cudaEvent_t` start/stop pair around the kernel-launch loop.
- CPU: `clock()` deltas around the serial loop.
- Final output reports both times and `cpu_time / gpu_time` as the speedup.

### Validation
After both implementations finish, the final GPU buffer (copied back to host as `comparedData`) is compared element-wise to `SerialPopulation`. A single mismatch sets `success = 1` and prints a failure message; otherwise the program confirms the CUDA implementation matches the reference.

## Dependencies

- CUDA toolkit (`nvcc`, `<cuda.h>`)
- OpenCV (`core`, `highgui`, `imgproc`)

## Notes

- `MAX_SIZE` is hard-capped at 1024 — larger grids will fail the `assert`.
- `srand(clock())` seeds the random initial state, so each run produces a different starting pattern.
- When `showtime > 0`, intermediate GPU states are copied back to host and displayed via OpenCV windows, which adds non-trivial overhead to the reported GPU time.
