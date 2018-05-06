#! /usr/bin/env python3.6
from argparse import ArgumentParser
from math import floor
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from openslide import OpenSlide
from skimage import morphology
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.filters.rank import median
from skimage.io import imsave
from skimage.transform import resize
import tensorflow as tf

from deeplab.utils import get_dataset_colormap


_OUTPUT_SEMANTIC_DIRNAME = 'semantic'
_OUTPUT_SOFTMAX_DIRNAME = 'softmax'
_OUTPUT_COLORED_SEMANTIC_DIRNAME = 'colored_semantic'
_OUTPUT_COLORED_SOFTMAX_DIRNAME = 'colored_softmax'
_WSI_FILES = '*.tif'
_PYPLOT_CMAP_NAME = 'jet'
_PATCH_SIDE = 1280
_PATCH_DIM = _PATCH_SIDE, _PATCH_SIDE
_SLIDE_LEVEL = 1
_SLIDE_THRESHOLD_LEVEL = 5
_PATCH_STRIDE = 1024

def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--wsi-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--pb-path',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--data-list-file',
        type=Path,
    )

    parser.add_argument(
        '--output-parent-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-folder-name',
        type=str,
        default='Predictions',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--save-colored',
        action='store_true',
    )

    return parser.parse_args()


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  OUTPUT_LOGITS_TENSOR_NAME = 'SoftmaxProbabilities:0'
  INPUT_SIZE = 1280
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, frozen_graph_filename):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    # Extract frozen graph from tar archive.
    with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, images):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    seg_maps, logits = self.sess.run(
        [self.OUTPUT_TENSOR_NAME, self.OUTPUT_LOGITS_TENSOR_NAME],
        feed_dict={self.INPUT_TENSOR_NAME: images})
    return images, seg_maps, logits


def get_names(data_list_file, wsi_dir):
    names = []
    if data_list_file is not None:
        with open(str(data_list_file)) as names_file:
            for name in names_file.readlines():
                names.append(name.strip())
    else:
        for wsi_file in wsi_dir.glob(_WSI_FILES):
            names.append(wsi_file.stem)
    return names


def threshold_hs(img_np):
    img_hsv = rgb2hsv(img_np)
    channel_h = img_hsv[:, :, 0]
    channel_s = img_hsv[:, :, 1]

    thresh_h = threshold_otsu(channel_h)
    thresh_s = threshold_otsu(channel_s)

    binary_h = channel_h > thresh_h
    binary_s = channel_s > thresh_s

    binary = np.bitwise_and(binary_h, binary_s)
    binary = morphology.remove_small_objects(binary)
    binary = morphology.remove_small_holes(binary)
    binary = median(binary)

    return binary.astype(bool)


def load_slide_threshold(slide, thresh_dim):
    w, h = slide.level_dimensions[_SLIDE_THRESHOLD_LEVEL]
    slide_img = (
        slide.read_region((0, 0), _SLIDE_THRESHOLD_LEVEL, (w, h))
        .convert('RGB'))
    thresh_h, thresh_w = thresh_dim
    w_min, h_min = min(w, thresh_w), min(h, thresh_h)
    thresh = np.full((thresh_h, thresh_w), False)
    thresh[:h_min, :w_min] = threshold_hs(slide_img)[:h_min, :w_min]
    return thresh


def generate_patch_batch(slide, threshold, thresh_patch_side, stride,
                         batch_size):
        coords = []
        images = []

        sample_level_factor = 2**_SLIDE_THRESHOLD_LEVEL
        h, w = threshold.shape
        h, w = h - stride, w - stride
        count = 0
        total = h * w // stride**2
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                region = threshold[y:(y + thresh_patch_side),
                                   x:(x + thresh_patch_side)]

                count += 1
                print(f'>> {(count / total * 100):.2f}% done', end='\r')

                if True not in region:
                    continue
                coord = x, y
                coord_0 = x * sample_level_factor, y * sample_level_factor
                patch = (slide.read_region(coord_0, _SLIDE_LEVEL, _PATCH_DIM)
                         .convert('RGB'))
                images.append(patch)
                coords.append(coord)


                if (len(images) == batch_size or
                        (y == h - stride and x == w - stride)):
                    yield np.stack(images), np.stack(coords)
                    coords = []
                    images = []


def main(args):
    output_dir = args.output_parent_dir / args.output_folder_name
    semantic_dir = output_dir / _OUTPUT_SEMANTIC_DIRNAME
    softmax_dir = output_dir / _OUTPUT_SOFTMAX_DIRNAME
    semantic_dir.mkdir(parents=True, exist_ok=True)
    softmax_dir.mkdir(parents=True, exist_ok=True)

    if args.save_colored:
        colored_semantic_dir = output_dir / _OUTPUT_COLORED_SEMANTIC_DIRNAME
        colored_softmax_dir = output_dir / _OUTPUT_COLORED_SOFTMAX_DIRNAME
        colored_semantic_dir.mkdir(parents=True, exist_ok=True)
        colored_softmax_dir.mkdir(parents=True, exist_ok=True)

    names = get_names(args.data_list_file, args.wsi_dir)

    deeplab_model = DeepLabModel(args.pb_path)
    cmap = plt.get_cmap(_PYPLOT_CMAP_NAME)

    name_count = len(names)
    for i, wsi_name in enumerate(names):
        wsi_path = args.wsi_dir / f'{wsi_name}.tif'
        slide = OpenSlide(str(wsi_path))

        print(f'>> Inferring {wsi_name} [{i + 1} / {name_count}]')

        w, h = slide.level_dimensions[_SLIDE_LEVEL]
        size_factor = 2**(_SLIDE_THRESHOLD_LEVEL - _SLIDE_LEVEL)
        thresh_patch_side = (_PATCH_SIDE // size_factor)
        thresh_patch_stride = (_PATCH_STRIDE // size_factor)
        thresh_patch_dim = thresh_patch_stride, thresh_patch_stride
        thresh_w = (floor(w / _PATCH_STRIDE) * thresh_patch_stride +
                    thresh_patch_side)
        thresh_h = (floor(h / _PATCH_STRIDE) * thresh_patch_stride +
                    thresh_patch_side)
        thresh_dim = thresh_h, thresh_w

        semantic_full = np.full(thresh_dim, 0, dtype=np.uint8)
        softmax_full = np.full(thresh_dim, 0, dtype=np.float32)

        threshold = load_slide_threshold(slide, thresh_dim)
        batch_size = args.batch_size
        patch_batches = generate_patch_batch(
            slide,
            threshold,
            thresh_patch_side,
            thresh_patch_stride,
            batch_size,
        )

        # iterations = 0
        # max_iterations = 1
        for images, coordinates in patch_batches:
            _, semantic, softmax = deeplab_model.run(images)

            for i in range(batch_size):
                x, y = coordinates[i]
                img = images[i]

                y_end = y + thresh_patch_stride
                x_end = x + thresh_patch_stride
                semantic_ds = resize(
                    semantic[i, :_PATCH_STRIDE, :_PATCH_STRIDE],
                    thresh_patch_dim,
                    preserve_range=True)
                semantic_full[y:y_end, x:x_end] = semantic_ds

                softmax_ds = resize(
                    softmax[i, :_PATCH_STRIDE, :_PATCH_STRIDE, 1],
                    thresh_patch_dim,
                )
                softmax_full[y:y_end, x:x_end] = softmax_ds

            # iterations += 1
            # if iterations > max_iterations:
            #     break

        orig_w, orig_h = slide.level_dimensions[_SLIDE_THRESHOLD_LEVEL]

        semantic_full = semantic_full[:orig_h, :orig_w]
        imsave(str(semantic_dir / f'{wsi_name}.png'), semantic_full)

        softmax_full = softmax_full[:orig_h, :orig_w]
        imsave(str(softmax_dir / f'{wsi_name}.png'), softmax_full)


        if args.save_colored:
            colored_semantic_full = get_dataset_colormap.label_to_color_image(
                semantic_full,
                get_dataset_colormap.get_camelyon_name(),
            )
            imsave(str(colored_semantic_dir / f'{wsi_name}.png'),
                   colored_semantic_full)

            colored_softmax_full = cmap(softmax_full)[:, :, :3]
            imsave(str(colored_softmax_dir / f'{wsi_name}.png'),
                   colored_softmax_full)

        slide.close()

if __name__ == '__main__':
    main(collect_arguments())
