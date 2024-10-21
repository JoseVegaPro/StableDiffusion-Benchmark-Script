# Benchmark Stable Diffusion (BSD)

This Python script, `bsd.py`, allows you to benchmark various Stable Diffusion models for generating images using GPU or CPU, with customizable options for precision, image resolution, and prompt inputs. It also logs system resource utilization (GPU, CPU, memory) during the image generation process.

## Features

- Select from multiple Stable Diffusion models, including:
  - Stable Diffusion v1.4
  - Stable Diffusion 2.1
  - Anything V3 (Anime Art)
  - Open Journey (MidJourney style)
- Customizable floating-point precision (16-bit, 32-bit, 8-bit quantization)
- Custom image resolutions (from 256x256 to 4K)
- Tracks GPU memory usage, GPU utilization, and CPU usage
- Benchmark multiple images with multi-threaded worker support
- Generates a detailed log file with performance metrics for each run

## Requirements

- Python 3.8+
- Required libraries:
  - `torch`
  - `diffusers`
  - `psutil`
  - `pynvml`
  - `bitsandbytes` (for 8-bit quantization)
  - `concurrent.futures`
  
You can install the dependencies by running:

```bash
pip install torch diffusers psutil pynvml bitsandbytes
