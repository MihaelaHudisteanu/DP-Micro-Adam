from collections.abc import Sequence
import dataclasses
import torch
import torchvision.transforms.functional as TF


@dataclasses.dataclass(kw_only=True, slots=True)
class AugmultConfig:
    ### CIFAR-10 / WRN-16-4 defaults (DeepMind JAX config):
    augmult: int = 4         
    random_crop: bool = True
    random_flip: bool = True
    random_color: bool = False
    pad: int | None = 4
    crop_size: Sequence[int] = (32, 32)
    #crop_size: Sequence[int] = (224, 224)  # ImageNet

    def apply(self, image, label):
        return apply_augmult(
            image, label,
            augmult=self.augmult,
            random_flip=self.random_flip,
            random_crop=self.random_crop,
            random_color=self.random_color,
            crop_size=self.crop_size,
            pad=self.pad,
        )


def apply_augmult(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    random_color: bool,
    crop_size: Sequence[int],
    pad: int | None,
):
    # ensure batch dim
    if images.dim() == 3:
        images = images.unsqueeze(0)             # (1,C,H,W)
        labels = labels.unsqueeze(0)

    B, C, H, W = images.shape
    device = images.device
    h, w = crop_size

    if augmult == 0:
        return images, labels

    out_imgs = torch.empty((augmult*B, C, h, w), dtype=images.dtype, device=device)
    out_labs = labels.repeat_interleave(augmult)

    for i in range(augmult):
        img_now = images.clone()

        if random_crop:
            if pad:
                img_now = padding_input(img_now, pad=pad)
            _, _, Hpad, Wpad = img_now.shape

            top  = torch.randint(0, Hpad - h + 1, (B,), device=device)
            left = torch.randint(0, Wpad - w + 1, (B,), device=device)
            cropped = [TF.crop(img_now[b], int(top[b]), int(left[b]), h, w) for b in range(B)]
            img_now = torch.stack(cropped, dim=0)

        if random_flip:
            mask = torch.rand(B, device=device) < 0.5
            img_now[mask] = torch.flip(img_now[mask], dims=[3])

        if random_color:
            jittered = []
            for b in range(B):
                im = img_now[b]
                im = TF.adjust_hue(im, float(torch.empty(1, device=device).uniform_(-0.1, 0.1)))
                im = TF.adjust_saturation(im, float(torch.empty(1, device=device).uniform_(0.6, 1.6)))
                im = TF.adjust_brightness(im, 1.0 + float(torch.empty(1, device=device).uniform_(-0.15, 0.15)))
                im = TF.adjust_contrast(im, float(torch.empty(1, device=device).uniform_(0.7, 1.3)))
                jittered.append(im)
            img_now = torch.stack(jittered, dim=0)

        out_imgs[i * B:(i + 1) * B] = img_now
        out_labs[i * B:(i + 1) * B] = labels

    return out_imgs, out_labs


def padding_input(x: torch.Tensor, pad: int):
    #mirror-pad on H and W.
    if x.dim() == 3:   # (C,H,W)
        x = torch.cat([x[:, :pad, :].flip(1), x, x[:, -pad:, :].flip(1)], dim=1)
        x = torch.cat([x[:, :, :pad].flip(2), x, x[:, :, -pad:].flip(2)], dim=2)
        return x
    if x.dim() == 4:   # (B,C,H,W)
        x = torch.cat([x[:, :, :pad, :].flip(2), x, x[:, :, -pad:, :].flip(2)], dim=2)
        x = torch.cat([x[:, :, :, :pad].flip(3), x, x[:, :, :, -pad:].flip(3)], dim=3)
        return x