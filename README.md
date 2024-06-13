# SD3-medium test

## Environment
Tested on `CUDA 12.1`, `diffuser 0.29.0`.
```
conda create -n sd3 python=3.10
conda activate sd3
pip install -r requirements.txt
```

```
# or install each package
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge diffusers
pip install transformers
pip install sentencepiece
pip install accelerate
pip install protobuf
```

## Run
```
python run.py
```

## Results
Using NVIDIA H100 80GB HBM3:
- SD2.1: VRAM 4GB, 1.8 secs
- SDXL: VRAM 9GB, 3.6 secs
- SD3-medium:  VRAM 18GB, secs
<img src='assets/sd3_comp.png'>