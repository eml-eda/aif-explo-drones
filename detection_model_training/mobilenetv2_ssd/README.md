# mobilenetv2_ssd

This folder contains two standalone `320x240` MobileNetV2 SSD detectors:

- `mobilenetv2_ssd.py`
  Standard PyTorch-style SSD. It uses the regular torchvision `SSD` training and inference behavior.
- `mobilenetv2_ssd_sigmoid.py`
  TF-inspired SSD with the same backbone and anchor layout, but a custom sigmoid classification path.

## Main difference

Both models share the same core detector structure:

- MobileNetV2 backbone
- 6 SSD feature maps
- `1x1` predictor heads
- reduced first feature-map anchors (`3, 6, 6, 6, 6, 6`)
- `ReLU6` in the extra SSD blocks by default
- input size `(320, 240)` in `(width, height)` order

If you want standard `ReLU` in the extra SSD blocks instead, pass `relu6=False` to either builder.

The main difference is the classification behavior:

- `mobilenet_v2_ssd(...)`
  Uses the default torchvision SSD classification semantics.
- `mobilenet_v2_ssd_sigmoid(...)`
  Uses BCE-with-logits during training and sigmoid scores at inference, with the internal background channel removed before NMS. This is the TensorFlow 1.x-style classification path.

## Params and MACs

For the comparable setup used against the paper target:

- standard model: `mobilenet_v2_ssd(num_classes=3)`
- sigmoid model: `mobilenet_v2_ssd_sigmoid(num_classes=2)`

Both have the same detector-core size and compute:

- Params: `4,627,367` (`4.63M`)
- MACs: `538,981,504` (`538.98 MMAC`)

Paper reference:

- Params: `4.67M`
- MACs: `534 MMAC`

Difference vs paper:

- Params: `42,633` fewer, absolute percentage difference `0.91%`
- MACs: `4,981,504` higher, absolute percentage difference `0.93%`

MACs here refer to the detector core only (`backbone + head`), excluding anchor generation and post-processing.

## Imports

```python
from mobilenetv2_ssd import mobilenet_v2_ssd
from mobilenetv2_ssd.mobilenetv2_ssd_sigmoid import mobilenet_v2_ssd_sigmoid
```

## Training Snippet

```python
import torch

from mobilenetv2_ssd import mobilenet_v2_ssd
# from mobilenetv2_ssd.mobilenetv2_ssd_sigmoid import mobilenet_v2_ssd_sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_classes means:
# - standard model: torchvision SSD convention
# - sigmoid model: number of foreground classes
model = mobilenet_v2_ssd(num_classes=3, relu6=True).to(device)
# model = mobilenet_v2_ssd(num_classes=3, relu6=False).to(device)
# model = mobilenet_v2_ssd_sigmoid(num_classes=2, relu6=True).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)

model.train()
for images, targets in train_loader:
    images = [img.to(device) for img in images]
    targets = [
        {
            "boxes": t["boxes"].to(device),
            "labels": t["labels"].to(device),
        }
        for t in targets
    ]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Utilities

```bash
.venv/bin/python mobilenetv2_ssd/profile_model.py
.venv/bin/python mobilenetv2_ssd/validate_models.py
```
