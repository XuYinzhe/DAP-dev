
## Installation

Clone the repo first:

```Bash
git clone https://github.com/XuYinzhe/DAP-dev.git
cd DAP
```

(Optional) Create a fresh conda env:

```Bash
conda create -n dap python=3.12
conda activate dap
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version, we test our code on torch==2.7.1 and torchvision==0.22.1)
pip install torch==2.7.1 torchvision==0.22.1

# other dependencies
pip install -r requirements.txt
```

In stall rasterizer:
```
pip install thirdparty/diff-gaussian-rasterization-omni --no-build-isolation
```

## Pre-trained model

Please download the pretrained model: https://huggingface.co/Insta360-Research/DAP-weights/blob/main/model.pth
Put it at `./weights/model.pth`

## Dataset

Please download from: https://hkust-vgd.nas.ust.hk:5001/sharing/7Ml9JHDpS

## Test training

```Bash
python CelestialSplat/train_simple.py --data_dir <downloaded dataset>
```
