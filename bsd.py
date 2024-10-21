import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import time
import os
import psutil
from concurrent.futures import ThreadPoolExecutor
import pynvml
from threading import Thread, Event

# Initialize NVML for GPU monitoring
pynvml.nvmlInit()

# Define the available models with descriptions
model_options = {
    '1': ('Stable Diffusion v1.4', 'CompVis/stable-diffusion-v1-4', 'The original stable diffusion model from CompVis.'),
    '2': ('Stable Diffusion 2.1', 'stabilityai/stable-diffusion-2-1', 'An enhanced and more recent version.'),
    '3': ('Anything V3', 'Linaqruf/anything-v3.0', 'A checkpoint for anime art generation.'),
    '4': ('Open Journey', 'prompthero/openjourney', 'Fine-tuned on images in the style of MidJourney.')
}

# Prompt the user to select a model
print("Choose the model to use:")
for key, (name, _, description) in model_options.items():
    print(f"{key}. {name}: {description}")
model_choice = input("Enter the number corresponding to your choice: ")

# Validate the user's choice
if model_choice in model_options:
    model_name, model_id, _ = model_options[model_choice]
    print(f"Using model: {model_name}")
else:
    print("Invalid choice. Defaulting to Stable Diffusion v1.4.")
    model_name, model_id, _ = model_options['1']

# Determine the device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the GPU type if using CUDA
if device == "cuda":
    gpu_type = torch.cuda.get_device_name(0)
else:
    gpu_type = "CPU"

# Prompt the user for floating-point precision
print("\nChoose floating-point precision for the model:")
print("1. 16-bit (torch.float16)")
print("2. 32-bit (torch.float32)")
print("3. 8-bit (int8 quantization with bitsandbytes)")
float_choice = int(input("Enter 1, 2, or 3: "))

if float_choice == 1:
    torch_dtype = torch.float16
    float_precision = "16-bit"
    model_load_kwargs = {'torch_dtype': torch_dtype}
elif float_choice == 2:
    torch_dtype = torch.float32
    float_precision = "32-bit"
    model_load_kwargs = {'torch_dtype': torch_dtype}
elif float_choice == 3:
    float_precision = "8-bit (int8)"
    model_load_kwargs = {'torch_dtype': torch.float16, 'load_in_8bit': True}
else:
    print("Invalid choice. Defaulting to 32-bit precision.")
    torch_dtype = torch.float32
    float_precision = "32-bit"
    model_load_kwargs = {'torch_dtype': torch_dtype}

# Prompt the user to select image resolution
print("\nChoose the image resolution:")
print("1. 256x256")
print("2. 512x512")
print("3. 1920x1080")
print("4. 3840x2160")
resolution_choice = int(input("Enter 1, 2, 3, or 4: "))

if resolution_choice == 1:
    height, width = 256, 256
elif resolution_choice == 2:
    height, width = 512, 512
elif resolution_choice == 3:
    height, width = 1080, 1920  # Note: Height first, then width
elif resolution_choice == 4:
    height, width = 2160, 3840
else:
    print("Invalid choice. Defaulting to 512x512.")
    height, width = 512, 512

# Assign run_id based on the resolution choice
if resolution_choice == 1:
    run_id = "Linode-RUN #1"
elif resolution_choice == 2:
    run_id = "Linode-RUN #2"
elif resolution_choice == 3:
    run_id = "Linode-RUN #3"
elif resolution_choice == 4:
    run_id = "Linode-RUN #4"
else:
    run_id = "Linode-RUN #Unknown"  # Default if something goes wrong

# Import bitsandbytes if using 8-bit precision
if float_choice == 3:
    import bitsandbytes as bnb

# Load the pipeline with the selected model and appropriate precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    **model_load_kwargs
)

# Adjust the scheduler based on the selected model
from diffusers import DPMSolverMultistepScheduler

if model_choice == '2':  # Stable Diffusion 2.1
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
else:
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Disable the safety checker (use responsibly)
pipe.safety_checker = None

# Move the model to the device
if device == 'cuda':
    pipe = pipe.to(device)

# Create output directory if it doesn't exist
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to get GPU memory and utilization using pynvml
def get_gpu_stats():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        memory_used = memory_info.used / (1024 ** 2)  # Convert bytes to MiB
        gpu_utilization = utilization.gpu  # In percent
        return memory_used, gpu_utilization
    except Exception as e:
        return f"Error: {str(e)}", None

# Function to monitor GPU utilization during inference
def monitor_gpu_utilization(utilization_list, stop_event):
    while not stop_event.is_set():
        _, gpu_utilization = get_gpu_stats()
        utilization_list.append(gpu_utilization)
        time.sleep(0.5)  # Adjust the sampling interval as needed

# Function to generate a single image (worker function)
def generate_image(worker_id, prompt, num_inference_steps, height, width, image_index):
    # Track metrics for each image
    latencies = []
    iteration_times = []
    memory_usages = []
    gpu_utilizations = []
    cpu_usages = []

    # Start timing for the latency (single image generation)
    image_start_time = time.time()

    # Synchronize GPU before starting
    if device == "cuda":
        torch.cuda.synchronize()

    # Start monitoring GPU utilization
    utilization_list = []
    stop_event = Event()
    monitor_thread = Thread(target=monitor_gpu_utilization, args=(utilization_list, stop_event))
    monitor_thread.start()

    # Generate the image with custom resolution
    if float_choice in [1, 2]:  # For 16-bit and 32-bit
        with autocast(device_type=device, dtype=torch_dtype):
            start_inference = time.time()
            images = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width).images
            if device == "cuda":
                torch.cuda.synchronize()  # Synchronize after inference
            end_inference = time.time()
    elif float_choice == 3:  # For 8-bit
        start_inference = time.time()
        images = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width).images
        if device == "cuda":
            torch.cuda.synchronize()
        end_inference = time.time()

    # Stop monitoring GPU utilization
    stop_event.set()
    monitor_thread.join()

    # Compute average GPU utilization during inference
    if utilization_list:
        avg_gpu_utilization = sum(utilization_list) / len(utilization_list)
    else:
        avg_gpu_utilization = 0.0

    # Stop timing for latency
    image_end_time = time.time()
    latencies.append((image_end_time - image_start_time) * 1000)  # Convert to milliseconds

    # Iteration time (for inference steps)
    iteration_times.append(end_inference - start_inference)

    # Save the image
    image_filename = f"worker_{worker_id}_image_{image_index}.png"
    images[0].save(os.path.join(output_dir, image_filename))

    # Capture and log GPU memory (after inference)
    memory_usage, _ = get_gpu_stats()
    memory_usages.append(memory_usage)
    gpu_utilizations.append(avg_gpu_utilization)

    # Capture CPU utilization
    cpu_usage = psutil.cpu_percent(interval=None)  # Get the current CPU usage
    cpu_usages.append(cpu_usage)

    print(f"Worker {worker_id} - Image {image_index + 1}: Memory usage: {memory_usage:.2f} MiB, "
          f"Average GPU Utilization: {avg_gpu_utilization:.2f}%, CPU Utilization: {cpu_usage}%")

    return {
        'latency': latencies,
        'iteration_times': iteration_times,
        'memory_usages': memory_usages,
        'gpu_utilizations': gpu_utilizations,
        'cpu_usages': cpu_usages
    }

# Function to run the benchmark with workers and log the results
def benchmark_stable_diffusion_with_workers(prompt, num_images=1, num_inference_steps=50, height=512, width=512, run_id=1, num_workers=1):
    latencies = []
    iteration_times = []
    memory_usages = []
    gpu_utilizations = []
    cpu_usages = []

    # Start timing for the entire run
    total_start_time = time.time()

    # Distribute images across workers using round-robin distribution
    image_assignments = [[] for _ in range(num_workers)]
    for i in range(num_images):
        worker_id = i % num_workers  # Round-robin assignment
        image_assignments[worker_id].append(i)

    # Start the worker pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id in range(num_workers):
            for image_index in image_assignments[worker_id]:
                futures.append(
                    executor.submit(generate_image, worker_id, prompt, num_inference_steps, height, width, image_index)
                )

        # Collect results
        for future in futures:
            result = future.result()
            latencies.extend(result['latency'])
            iteration_times.extend(result['iteration_times'])
            memory_usages.extend(result['memory_usages'])
            gpu_utilizations.extend(result['gpu_utilizations'])
            cpu_usages.extend(result['cpu_usages'])

    # Stop timing for the entire run
    total_time = time.time() - total_start_time

    # Calculate the metrics
    avg_latency = sum(latencies) / len(latencies)
    throughput = num_images / total_time
    avg_iteration_speed = sum(iteration_times) / len(iteration_times)
    iterations_per_second = num_inference_steps / avg_iteration_speed  # Calculate iterations per second
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    print(f"\nGenerated {num_images} image(s) in {total_time:.2f} seconds")
    print(f"Average Latency: {avg_latency:.2f} ms per image")
    print(f"Throughput: {throughput:.2f} images per second")
    print(f"Average Iteration Speed: {avg_iteration_speed:.2f} seconds per iteration")
    print(f"Iterations per second: {iterations_per_second:.2f} it/s")
    print(f"Average CPU Utilization: {avg_cpu_usage:.2f}%")
    print(f"Average GPU Utilization: {avg_gpu_utilization:.2f}%")
    print(f"Average GPU Memory Usage: {avg_memory_usage:.2f} MiB")

    # Write log file with all metrics
    log_file = "benchmark_SD_log.txt"
    with open(log_file, "a") as f:
        f.write(f"{run_id} - Image size: {height}x{width}\n")  # Updated format here
        f.write(f"Model used: {model_name} ({model_id})\n")
        f.write(f"GPU Type: {gpu_type}\n")
        f.write(f"Floating-Point Precision: {float_precision}\n")
        f.write(f"Total Time: {total_time:.2f} seconds\n")
        f.write(f"Average Latency: {avg_latency:.2f} ms per image\n")
        f.write(f"Throughput: {throughput:.2f} images per second\n")
        f.write(f"Average Iteration Speed: {avg_iteration_speed:.2f} seconds per iteration\n")
        f.write(f"Iterations per second: {iterations_per_second:.2f} it/s\n")
        f.write(f"Inference steps (iterations): {num_inference_steps}\n")
        f.write(f"Number of images generated: {num_images}\n")
        f.write(f"Total Workers: {num_workers}\n")
        f.write(f"Average CPU Utilization: {avg_cpu_usage:.2f}%\n")
        f.write(f"Average GPU Utilization: {avg_gpu_utilization:.2f}%\n")
        f.write(f"Average GPU Memory Usage: {avg_memory_usage:.2f} MiB\n")
        f.write(f"Prompt used: {prompt}\n")
        f.write("=============================================\n")

# Define your prompt with default option
user_prompt = input("\nEnter the prompt for image generation (press Enter to use the default prompt):\n")
if user_prompt.strip() == '':
    prompt = "A scenic mountain landscape with a clear blue sky."
    print(f"Using default prompt: {prompt}")
else:
    prompt = user_prompt

# Prompt the user for number of images, number of workers, and number of iterations
num_images = int(input("\nEnter the number of images to generate (e.g., 5): "))
num_inference_steps = int(input("Enter the number of inference steps (iterations) (e.g., 50): "))

# Ensure that the number of inference steps does not exceed 1000
if num_inference_steps > 1000:
    print(f"Number of inference steps exceeds 1000. Setting to 1000.")
    num_inference_steps = 1000

num_workers = int(input("Enter the number of workers (e.g., 2): "))  # Prompt for the number of workers

# Run the benchmark with user-defined settings
benchmark_stable_diffusion_with_workers(
    prompt,
    num_images,
    num_inference_steps,
    height=height,
    width=width,
    run_id=run_id,
    num_workers=num_workers
)

# Shutdown NVML
pynvml.nvmlShutdown()
