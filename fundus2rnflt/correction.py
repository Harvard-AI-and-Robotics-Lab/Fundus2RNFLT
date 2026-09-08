"""RNFLT map + mask -> artifact-corrected map, by inpainting holes with the RNFLT2Vec model
(TensorFlow 2.4). RNFLT2Vec itself comes from the git submodule third_party/RNFLT2Vec."""
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

IMG_SIZE = 256  # RNFLT2Vec input size
MORPH_KERNEL_HW = (4, 4)
THRESHOLD_UM = 30.0  # tissue thinner than this (outside disc/cup) is treated as a hole
RNFLT_MAX_UM = 350.0  # scale RNFLT2Vec was trained with
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_HW)

DEFAULT_RNFLT2VEC_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'RNFLT2Vec')

# Constructor arguments RNFLT2Vec was published with (only those construct_model_from_args reads).
RNFLT2VEC_ARGS = dict(
    runmodel="RNFLT2vec",
    img_rows=IMG_SIZE, img_cols=IMG_SIZE,
    projection_dim=128, embed_dim=512, projection_layers=3,
    reconstruct_w=1.0, contrastive_w=0.001, consistency_w=0.04,
    batch_size=4,
)


class InpaintPredictor:
    """The RNFLT2Vec inpainting head and the upstream colorizer (to_3dmap_v2) its inputs need."""

    def __init__(self, keras_model, to_3dmap_v2):
        self.keras_model = keras_model
        self.to_3dmap_v2 = to_3dmap_v2


def load_inpaint_predictor(weights_path, vgg_weights, rnflt2vec_repo):
    """Build RNFLT2Vec from the submodule, load its weights, return its inpaint_model head.

    weights_path: combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5 (RNFLT2Vec parses
        the epoch from the file name, so keep it).
    vgg_weights: 'imagenet' or a VGG16 .h5; VGG16 only enters RNFLT2Vec's training loss and does
        not affect the inpainting output.
    """
    assert os.path.isfile(os.path.join(rnflt2vec_repo, 'models', 'rnflt2vec.py')), (
        f"RNFLT2Vec not found at {rnflt2vec_repo}; run 'git submodule update --init' "
        "or pass --rnflt2vec-repo")
    assert os.path.isfile(weights_path), f"Missing RNFLT2Vec weights: {weights_path}"
    # RNFLT2Vec's load() reads the epoch from the file name: <name>.<epoch>-<loss>.h5
    epoch_field = os.path.basename(weights_path).split('.')[1].split('-')[0]
    assert epoch_field.isdigit() and int(epoch_field) > 0, (
        f"RNFLT2Vec parses the epoch from the weights file name; keep the original name "
        f"(e.g. combined_rnflt2vec_weights_512_128_10_0001_004.93-0.03.h5), got {os.path.basename(weights_path)}")
    # The submodule's top-level packages are named 'models' and 'utils'; refuse to shadow or be shadowed.
    assert 'models' not in sys.modules and 'utils' not in sys.modules, (
        "a top-level 'models' or 'utils' module is already imported; RNFLT2Vec cannot be loaded into this process")
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    sys.path.insert(0, rnflt2vec_repo)
    from models import rnflt2vec
    from utils.map_handler import to_3dmap_v2

    rnflt2vec_model = rnflt2vec.construct_model_from_args(Namespace(vgg_weights=vgg_weights, **RNFLT2VEC_ARGS))
    rnflt2vec_model.load(weights_path, train_bn=False, lr=0.00005)
    inpaint_layer = rnflt2vec_model.model.get_layer('inpaint_model')
    keras_model = Model(inputs=inpaint_layer.inputs, outputs=inpaint_layer.outputs)
    return InpaintPredictor(keras_model, to_3dmap_v2)


def preprocess_to_inpaint_inputs(rnflt_um_224, mask_codes_224, threshold_um, to_3dmap_v2):
    """RNFLT map (µm, NaN outside tissue) + mask codes (0 invalid, 1 tissue, 2 disc, 3 cup) ->
    RNFLT2Vec inputs at 256x256.

    Returns:
      masked_map: float32 [1,256,256,3] in [0,1], grey-colorized thickness, holes set to white
      ori_mask: uint8 [1,256,256,3], 1 = keep, 0 = hole (invalid or thin, never disc/cup)
      r256_closed: float32 [256,256], resized and morphologically closed thickness
    """
    # 1) resize to 256
    r256 = cv2.resize(rnflt_um_224.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    # 2) morph close (fill NaNs with 0 first)
    r256_filled = np.nan_to_num(r256, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    r256_closed = cv2.morphologyEx(r256_filled, cv2.MORPH_CLOSE, MORPH_KERNEL)
    # 3) ori_mask (3-channel, uint8); invalid/thin=0, disc/cup forced 1
    m256_codes = cv2.resize(mask_codes_224.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    ori_mask = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    invalid = (m256_codes == 0)
    disc = (m256_codes == 2)
    cup = (m256_codes == 3)
    thin = (r256_closed < float(threshold_um))
    holes = (invalid | thin) & (~disc) & (~cup)
    ori_mask[holes] = (0, 0, 0)
    # 4) colorize with to_3dmap_v2 (RNFLT2Vec) and scale to [0,1]; np.clip gives it a fresh copy
    #    because to_3dmap_v2 modifies its input in place
    rgb_uint8 = to_3dmap_v2(np.clip(r256_closed, 0, RNFLT_MAX_UM))
    masked_map = (rgb_uint8.astype(np.float32) / 255.0)
    # 5) holes -> white
    masked_map[ori_mask == 0] = 1.0
    return masked_map[None, ...], ori_mask[None, ...], r256_closed


def predict_corrected_map(predictor, rnflt_um_224, mask_codes_224, threshold_um):
    """Inpainted thickness map (µm, float32) at the input map's resolution, everywhere."""
    H0, W0 = rnflt_um_224.shape
    masked_map, ori_mask, _ = preprocess_to_inpaint_inputs(rnflt_um_224, mask_codes_224, threshold_um, predictor.to_3dmap_v2)
    # inpaint predict (returns 3-channel RGB in [0,1])
    pred_rgb = predictor.keras_model.predict([masked_map, ori_mask], verbose=0)[0]
    # convert to µm via mean over channels * 350 (RNFLT2Vec)
    pred_um_256 = np.mean(np.clip(pred_rgb, 0.0, 1.0), axis=2) * RNFLT_MAX_UM
    return cv2.resize(pred_um_256, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def blend_corrected(original_um_224, corrected_um_224, mask_codes_224, threshold_um):
    """Original map with the inpainted values pasted in at the hole pixels only
    (invalid or thinner than threshold_um, never disc/cup)."""
    out = original_um_224.copy()
    disc = (mask_codes_224 == 2)
    cup = (mask_codes_224 == 3)
    thin = (np.nan_to_num(original_um_224, nan=0.0) < float(threshold_um))
    replace = ((mask_codes_224 == 0) | thin) & (~disc) & (~cup)
    out[replace] = corrected_um_224[replace]
    return out.astype(np.float32)
