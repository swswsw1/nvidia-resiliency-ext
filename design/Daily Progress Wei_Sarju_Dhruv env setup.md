# **NVRx Development Environment Setup**

## **Prerequisites**

* Docker with NVIDIA GPU support (nvidia-docker or \--gpus flag)  
* Access to NVIDIA NGC container registry (nvcr.io)

## **Steps**

### **1\. Clone the repo on your host machine**

cd /raid/wei23/sarju   \# or your preferred working directory  
git clone https://github.com/NVIDIA/nvidia-resiliency-ext.git

### **2\. Start the container**

Use the `pytorch:24.12-py3` container — this is the version the NVRx team tests against. Newer containers (e.g., 25.04) have protobuf version conflicts that break the build.

docker run \-dit \\  
  \--name sarju\_nvrx\_root \\  
  \--gpus all \\  
  \-v /raid/wei23/sarju:/workspace \\  
  \-w /workspace \\  
  nvcr.io/nvidia/pytorch:24.12-py3 \\  
  bash

**Note:** Do not use `--user $(id -u):$(id -g)`. Running as root inside the container is required for `pip install` to write to system site-packages.

### **3\. Enter the container**

docker exec \-it sarju\_nvrx\_root bash

### **4\. Fix PyTorch for B200 GPUs**

The `pytorch:24.12-py3` container ships with PyTorch built for CUDA 13.0 (`torch 2.11.0+cu130`), but the NVIDIA driver in the container only supports CUDA 12.8. On B200 GPUs (compute capability `sm_100`), this results in a driver version mismatch error.

Downgrade to the CUDA 12.8 build of PyTorch, which includes `sm_100` support and is compatible with the driver:

pip install torch==2.11.0+cu128 \--index-url https://download.pytorch.org/whl/cu128  
pip install torchvision \--index-url https://download.pytorch.org/whl/cu128

Verify the fix:

python \-c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.get\_arch\_list()); print(torch.cuda.is\_available())"

You should see `sm_100` in the arch list and `True` for CUDA availability.

### **5\. Fix git ownership and install**

The repo was cloned as your host user but you're root inside the container. Git will refuse to operate on it without this:

git config \--global \--add safe.directory /workspace/nvidia-resiliency-ext  
cd /workspace/nvidia-resiliency-ext  
pip install \-e .

### **6\. Pre-download MNIST dataset (for examples)**

The straggler detection example uses MNIST. When running with multiple GPUs, all processes try to download the dataset simultaneously, which can cause race conditions and corrupted files. Pre-download it once before running:

python \-c "from torchvision import datasets; datasets.MNIST('data', train=True, download=True)"

This only needs to be done once — subsequent runs use the cached data.

### **7\. Verify the installation**

python \-c "from nvidia\_resiliency\_ext.attribution.trace\_analyzer.fr\_attribution import CollectiveAnalyzer; print('works')"

Run an example:

cd /workspace/nvidia-resiliency-ext  
python examples/straggler/example.py

## **Re-entering the container**

If you exit the container, it keeps running in the background. Just re-enter:

docker exec \-it sarju\_nvrx\_root bash

If the container was stopped (e.g., after a reboot):

docker start sarju\_nvrx\_root  
docker exec \-it sarju\_nvrx\_root bash

## **Development workflow**

Since the package is installed with `-e` (editable mode), any changes you make to files under `src/` take effect immediately — no need to reinstall. Edit files on your host machine with your preferred editor; the changes are visible inside the container via the volume mount.

The FR attribution code is at:

src/nvidia\_resiliency\_ext/attribution/trace\_analyzer/fr\_attribution.py

## **Troubleshooting**

| Problem | Fix |
| ----- | ----- |
| `NVIDIA B200 with CUDA capability sm_100 is not compatible` | Run `pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128` (see step 4\) |
| `CUDA initialization: The NVIDIA driver on your system is too old` | You have the `cu130` build of PyTorch but a CUDA 12.8 driver — install the `cu128` build instead (see step 4\) |
| `torchvision` / `operator torchvision::nms does not exist` | Run `pip install torchvision --index-url https://download.pytorch.org/whl/cu128` to get a matching version |
| MNIST download fails with `File not found or corrupted` | Pre-download with a single process first (see step 6\) |
| `pip install` fails with protobuf conflict | Make sure you're using `pytorch:24.12-py3`, not 25.04 |
| `pip install` fails with permission errors | Make sure you're running as root (no `--user` flag in `docker run`) |
| `pip install` fails with "dubious ownership" | Run `git config --global --add safe.directory /workspace/nvidia-resiliency-ext` |
| `ModuleNotFoundError: nvidia_resiliency_ext` | Either run `pip install -e .` or set `export PYTHONPATH=/workspace/nvidia-resiliency-ext/src:$PYTHONPATH` |

