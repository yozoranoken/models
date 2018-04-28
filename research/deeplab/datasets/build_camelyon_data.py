# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts CAMELYON Patch data to TFRecord file format with Example protos.

CAMELYON Patch dataset is expected to have the following directory structure:

    + Patches
        + data
        + labels
        + tfrecord

Data folder:
    $CWD/data (default)

Labels folder:
    $CWD/labels (default)

List folder:
    $CWD (default)

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

    image/encoded: encoded image content.
    image/filename: image filename.
    image/format: image file format.
    image/height: image height.
    image/width: image width.
    image/channels: image channels.
    image/segmentation/class/encoded: encoded semantic segmentation content.
    image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os
import os.path
import sys

import tensorflow as tf

import build_data


_CWD = os.getcwd()


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_folder',
                           os.path.join(_CWD, 'data'),
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'labels_folder',
    os.path.join(_CWD, 'labels'),
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_file',
    '',
    'File containing list of filenames to include in tfrecords.')

tf.app.flags.DEFINE_string(
    'output_dir',
    os.path.join(_CWD, 'tfrecord'),
    'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_integer(
    'num_shards',
    8,
    'Numer of shards to split tfrecords.')


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
        dataset_split: The dataset split (e.g., train, test).

    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(FLAGS.num_shards)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(FLAGS.num_shards):
        output_filename = os.path.join(
                FLAGS.output_dir,
                '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id,
                                              FLAGS.num_shards))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i + 1, len(filenames), shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = os.path.join(
                        FLAGS.data_folder,
                    filenames[i] + '.' + FLAGS.image_format)
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = os.path.join(
                        FLAGS.labels_folder,
                        filenames[i] + '.' + FLAGS.label_format)
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError(
                        'Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                        image_data, filenames[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    _convert_dataset(FLAGS.list_file)


if __name__ == '__main__':
    tf.app.run()
