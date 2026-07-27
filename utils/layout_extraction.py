from typing import Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, label

try:
    from groundingdino.util import box_ops
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
except ImportError as exc:
    raise ImportError(
        "GroundingDINO is required for layout extraction. Install it with "
        "`pip install git+https://github.com/IDEA-Research/GroundingDINO.git`."
    ) from exc


def _load_image_dino(image_source) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict_mask(segmentmodel, sam, image, text_prompt, cnt, confidence=0.2, threshold=0.5):
    image_source, image = _load_image_dino(image)
    boxes, logits, _ = predict(
        model=segmentmodel,
        image=image,
        caption=text_prompt,
        box_threshold=0.3,
        text_threshold=0.25,
    )

    if boxes.shape[0] < cnt:
        raise RuntimeError(f"Only {boxes.shape[0]} boxes are detected, But {cnt} boxes are required.")
    topk_indices = torch.topk(logits, cnt)[1]
    boxes = boxes[topk_indices]

    sam.set_image(image_source)
    h, w, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
    transformed_boxes = sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.squeeze(1)


def save_mask_tensor_as_image(tensor: torch.Tensor, file_path: str):
    if tensor.dtype not in [torch.float32, torch.float64, torch.bool]:
        raise ValueError

    if tensor.dtype == torch.bool:
        tensor = tensor.float()

    if tensor.dim() == 3 and tensor.size(2) == 1:
        tensor = tensor.squeeze(2)
    elif tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    elif tensor.dim() != 2:
        raise ValueError
    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError

    tensor_uint8 = (tensor * 255).to(dtype=torch.uint8)
    array = tensor_uint8.cpu().numpy()
    image = Image.fromarray(array, mode="L")
    image.save(file_path)


def _retain_largest_component(mask):
    mask_np = mask.cpu().numpy()
    labeled_array, num_features = label(mask_np)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    largest_component_mask = labeled_array == largest_label
    return torch.tensor(largest_component_mask, dtype=torch.bool, device=mask.device)


def _calculate_mask_area(mask):
    return mask.sum().item()


def _dilate_mask(mask, k):
    mask_np = mask.cpu().numpy()
    structure = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    dilated_mask_np = binary_dilation(mask_np, structure=structure)
    return torch.tensor(dilated_mask_np, dtype=torch.bool, device=mask.device)


def process_masks(mask_list):
    indexed_masks = [(i, mask, _calculate_mask_area(mask)) for i, mask in enumerate(mask_list)]
    sorted_masks = sorted(indexed_masks, key=lambda x: x[2])

    covered_regions = torch.zeros_like(mask_list[0], dtype=torch.bool)
    pcs_mask_list = [None] * len(mask_list)

    for original_index, mask, _ in sorted_masks:
        non_overlapping_mask = mask & ~covered_regions
        largest_cc_mask = _retain_largest_component(non_overlapping_mask)
        largest_cc_mask = _dilate_mask(largest_cc_mask, 10)
        largest_cc_mask = largest_cc_mask & ~covered_regions
        pcs_mask_list[original_index] = largest_cc_mask
        covered_regions |= largest_cc_mask

    return pcs_mask_list
