# CLIP

在程序中调用 CLIP 预训练模型

```bash
cd ./experiment
```

```python
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

Backbone_name = ["RN50", "RN101", "RN50x4", "RN50x16",
                 "RN50x64", "ViT-B/32", "ViT-B/16",
                 "ViT-L/14", "ViT-L/14@336px"]

model, preprocess = clip.load("Backbone name", device=device)

for item in data:
    image, label = item

    image_features = model.encode_img(image)
    text_features = model.encode_text(text)

    logit_per_image, logit_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

```
