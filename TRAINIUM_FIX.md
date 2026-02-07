# Trainium Device Detection Fix

## Problem
The MNIST training script was using CPU instead of Trainium accelerator, showing:
```
Using CPU
```

## Root Causes

### 1. Missing Neuron Kernel Driver
**Issue**: The `aws-neuronx-dkms` kernel module was not installed
**Symptom**: `neuron-ls` failed with "no neuron device found"
**Fix**: Installed the Neuron DKMS driver

### 2. Wrong torch-neuronx Package
**Issue**: Installed placeholder package from PyPI instead of official AWS package
**Symptom**: `ImportError: WRONG PACKAGE. Please install from pip.repos.neuron.amazonaws.com`
**Fix**: Reinstalled from official Neuron pip repository

### 3. Incorrect Device Detection Code
**Issue**: Code used `torch.accelerator.is_available()` which doesn't detect XLA/Neuron
**Symptom**: `torch.accelerator.is_available()` returned False despite devices being present
**Fix**: Updated code to use `torch_xla.device()` for Neuron device detection

## Solution Steps

### Step 1: Install Neuron Kernel Driver
```bash
# Update package lists
sudo apt-get update -y

# Install kernel headers (required for DKMS compilation)
sudo apt-get install linux-headers-$(uname -r) -y

# Install Neuron kernel driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Verify driver loaded
lsmod | grep neuron
ls -la /dev/neuron*
neuron-ls
```

**Result**:
- Neuron module loaded: `neuron 462848 0`
- Device file created: `/dev/neuron0`
- Hardware detected: `trn1.2xlarge` with 2 NeuronCores and 32GB memory

### Step 2: Install Correct torch-neuronx Package
```bash
# Uninstall wrong package
pip uninstall torch-neuronx -y

# Configure pip to use Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install correct packages
pip install neuronx-cc==2.* torch-neuronx torchvision
```

**Result**:
- neuronx-cc: 2.22.12471.0 (Neuron compiler)
- torch-neuronx: 2.9.0.2.11.19912
- torch: 2.9.1 (downgraded from 2.10.0 for compatibility)
- torch-xla: 2.9.0
- libneuronxla: 2.2.14584.0

### Step 3: Update Code for XLA Backend Detection
Modified `main.py` to use XLA device API instead of torch.accelerator:

**Before**:
```python
import torch
use_accel = not args.no_accel and torch.accelerator.is_available()
if use_accel:
    device = torch.accelerator.current_accelerator()
```

**After**:
```python
import torch
try:
    import torch_neuronx
    import torch_xla
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False

use_accel = not args.no_accel and NEURON_AVAILABLE
if use_accel:
    device = torch_xla.device()  # Returns xla:0
    device_type = 'xla'
```

Also updated:
- Device type checks: `'neuron'` → `'xla'`
- Synchronization: `xm.mark_step()` → `torch_xla.sync()`

## Verification

### Before Fix:
```
Using CPU
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.329474
```

### After Fix:
```
Using Neuron device: xla:0 (type: xla)
Compiler status PASS
2026-02-07 22:23:02: Compilation Successfully Completed for model.MODULE_17530823641372004936
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.308204
```

## Run Training
```bash
cd /home/ubuntu/aws-torchneuron/1_mnist
source .venv/bin/activate
python main.py --epochs 3
```

## Key Learnings

1. **TorchNeuron uses XLA backend**: Device detection must use `torch_xla.device()`, not `torch.accelerator`
2. **Driver installation is critical**: Without `aws-neuronx-dkms`, no `/dev/neuron*` devices exist
3. **Use official packages**: The Neuron pip repository is required, not PyPI placeholders
4. **XLA device type is 'xla'**: Code must reference `'xla'` not `'neuron'` for device type checks

## Hardware Detected
- Instance: `trn1.2xlarge`
- NeuronCores: 2
- Memory: 32 GB
- PCI Device: `0000:00:1e.0`
- Devices: `xla:0`, `xla:1`
