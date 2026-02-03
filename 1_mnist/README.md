# TorchNeuron MNIST Training Example

This example demonstrates how to train a CNN on the MNIST dataset using TorchNeuron on AWS Trainium instances.

## Overview

TorchNeuron is an open-source PyTorch backend that provides native PyTorch framework integration for AWS Trainium. This example showcases:

- **Automatic Device Detection**: Uses `torch.accelerator` API for seamless device detection
- **torch.compile Optimization**: JIT compilation with `backend='neuron'` for improved performance
- **Mixed Precision Training**: Uses `torch.autocast(device_type='neuron')` for automatic datatype conversion

## Requirements

### On Trainium Instance (trn1, trn2, etc.)

```bash
# Activate your PyTorch virtual environment
source ~/aws_neuron_venv_pytorch/bin/activate

# Install PyTorch NeuronX (if not already installed)
pip install torch-neuronx torchvision

# Verify installation
python -c "import torch; print(f'Accelerator available: {torch.accelerator.is_available()}')"
```

### For CPU Testing

```bash
pip install torch torchvision
```

## Usage

### Basic Training (Eager Mode)

```bash
# On Trainium - uses automatic device detection
python main.py --epochs 3

# On CPU (for testing)
python main.py --no-accel --epochs 3
```

### With torch.compile Optimization

```bash
# Enable JIT compilation for improved performance
python main.py --epochs 3 --compile
```

### With Mixed Precision

```bash
# Enable automatic mixed precision training
python main.py --epochs 3 --compile --mixed-precision
```

### Quick Validation (Dry Run)

```bash
# Run single batch to verify setup
python main.py --dry-run --epochs 1
```

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 64 | Training batch size |
| `--test-batch-size` | 1000 | Test batch size |
| `--epochs` | 14 | Number of training epochs |
| `--lr` | 1.0 | Learning rate |
| `--gamma` | 0.7 | Learning rate decay factor |
| `--no-accel` | False | Disable accelerator (use CPU) |
| `--dry-run` | False | Run single batch only |
| `--seed` | 1 | Random seed |
| `--log-interval` | 10 | Logging frequency (batches) |
| `--save-model` | False | Save trained model |
| `--compile` | False | Enable torch.compile optimization |
| `--mixed-precision` | False | Enable mixed precision training |

## TorchNeuron Key Concepts

### Device Placement

TorchNeuron registers the `'neuron'` device type. The code uses `torch.accelerator` API for automatic detection:

```python
# Automatic detection (works for CUDA, Neuron, etc.)
device = torch.accelerator.current_accelerator()

# Or explicit placement
device = torch.device('neuron')
model = Net().to(device)
```

### torch.compile with Neuron Backend

For JIT compilation on Trainium:

```python
model = torch.compile(model, backend='neuron')
```

### Mixed Precision

Use standard PyTorch autocast with the 'neuron' device type:

```python
with torch.autocast(device_type='neuron'):
    output = model(data)
    loss = F.nll_loss(output, target)
```

## Performance Tips

1. **Use torch.compile**: The Neuron backend optimizes the computation graph for Trainium hardware
2. **Enable Mixed Precision**: Reduces memory usage and can improve throughput
3. **Batch Size**: Experiment with larger batch sizes to better utilize NeuronCores
4. **Data Loading**: Use `pin_memory=True` and `persistent_workers=True` for efficient data loading

## Expected Output

After training, you should see output similar to:

```
Using accelerator: neuron:0 (type: neuron)
Applying torch.compile with backend='neuron'
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.302585
...
Test set: Average loss: 0.0456, Accuracy: 9859/10000 (99%)

Total time: 0:02:15 (135.234 seconds)
```

## References

- [TorchNeuron Overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html)
- [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
- [TorchNeuron GitHub](https://github.com/aws-neuron/torch-neuronx)
