#! /usr/bin/env python3.6
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
import os
from io import BytesIO
import tarfile
import tempfile
import sys

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread

import tensorflow as tf

from deeplab import common
from deeplab.datasets import build_data


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--image-dir',
        type=Path,
        required=True,
        metavar='IMAGE_DIR',
    )

    parser.add_argument(
        '--pb-path',
        type=Path,
        required=True,
        metavar='PB_PATH',
    )

    return parser.parse_args()


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  OUTPUT_LOGITS_TENSOR_NAME = 'SemanticProbabilities:0'
  INPUT_SIZE = 768
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
    seg_maps, logits = self.sess.run([self.OUTPUT_TENSOR_NAME, self.OUTPUT_LOGITS_TENSOR_NAME],
                             feed_dict={self.INPUT_TENSOR_NAME: images})
    return images, seg_maps, logits


def read_images(image_dir):
    images = []
    image_reader = build_data.ImageReader('jpeg', channels=3)
    for image_path in image_dir.glob('*.jpeg'):
        image_data = tf.gfile.FastGFile(str(image_path), 'rb').read()
        image = image_reader.decode_image(image_data)
        images.append(image)
    shuffle(images)
    return np.stack(images)


def main(args):
    images = read_images(args.image_dir)

    deeplab_model = DeepLabModel(args.pb_path)
    for s in range(0, images.shape[0], 2):
        batch_images, predictions, logits = (
            deeplab_model.run(images[s:s+2, :, :, :]))

        f, ax = plt.subplots(nrows=batch_images.shape[0], ncols=3)
        for i in range(batch_images.shape[0]):
            lgs = logits[i, : , :, 1]
            print(lgs[0][0])
            ax[i][0].imshow(batch_images[i])
            ax[i][1].imshow(predictions[i])
            ax[i][2].imshow(lgs)
        plt.show()


if __name__ == '__main__':
    main(collect_arguments())
