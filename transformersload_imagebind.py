"""
加载  ImageBind  检索模型

pip install torch torchvision torchaudio
pip install ftfy regex

"""


import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# 加载模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# 或者加载本地权重
# model.load_state_dict(torch.load("/localmodels/ImageBind/imagebind_huge.pth"))