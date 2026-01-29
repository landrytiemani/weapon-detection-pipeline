# Installation Guide

## Requirements
- Python 3.10+
- CUDA 11.8+ (GPU)
- 8GB+ GPU VRAM

## Steps

1. Clone repository
```bash
git clone https://github.com/landrytiemani/weapon-detection-pipeline.git
cd weapon-detection-pipeline
```

2. Create environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Download weights
```bash
bash scripts/download_weights.sh
```

## Verify
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
