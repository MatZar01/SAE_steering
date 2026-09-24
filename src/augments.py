"""Stochastic training transforms and deterministic evaluation/export transforms."""

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from cfgs.cfg import AugmentConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(config: AugmentConfig, training: bool = False) -> T.Compose:
    if training:
        operations = [
            T.RandomResizedCrop(config.image_size, scale=config.crop_scale,
                                interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(config.horizontal_flip_probability),
            T.RandomVerticalFlip(config.vertical_flip_probability),
            T.RandomRotation(config.rotation_degrees),
            T.ColorJitter(brightness=config.color_jitter, contrast=config.color_jitter,
                          saturation=config.color_jitter),
            T.RandomApply([T.GaussianBlur(config.gaussian_blur_kernel,
                                         sigma=config.gaussian_blur_sigma)],
                          p=config.gaussian_blur_probability),
        ]
    else:
        operations = [T.Resize(round(config.image_size * 256 / 224)),
                      T.CenterCrop(config.image_size)]
    operations += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if training:
        # Zero in normalized space masks a patch with the ImageNet mean color.
        operations.append(T.RandomErasing(p=config.masking_probability,
                                           scale=config.masking_scale, value=0))
    return T.Compose(operations)
