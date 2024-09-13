import torch
from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
fig_, ax_ = metric.plot()
