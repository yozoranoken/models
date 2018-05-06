# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import os
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from deeplab import common
from deeplab import input_preprocess
from deeplab import model

slim = tf.contrib.slim
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')

flags.DEFINE_string('export_path', None,
                    'Path to output Tensorflow frozen graph.')

flags.DEFINE_integer('num_classes', 2, 'Number of classes.')

flags.DEFINE_multi_integer('crop_size', [768, 768],
                           'Crop size [height, width].')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale inference.
flags.DEFINE_multi_float('inference_scales', [1.0],
                         'The scales to resize images for inference.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images during inference or not.')

# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported model.
_OUTPUT_NAME = 'SemanticPredictions'
_OUTPUT_SOFTMAX_NAME = 'SoftmaxProbabilities'


def _create_input_tensors():
  """Creates and prepares input tensors for DeepLab model.

  This method creates a 4-D uint8 image tensor 'ImageTensor' with shape
  [1, None, None, 3]. The actual input tensor name to use during inference is
  'ImageTensor:0'.

  Returns:
    image: Preprocessed 4-D float32 tensor with shape [1, crop_height,
      crop_width, 3].
    original_image_size: Original image shape tensor [height, width].
    resized_image_size: Resized image shape tensor [height, width].
  """
  # input_preprocess takes 4-D image tensor as input.
  tensor_shape = [None, FLAGS.crop_size[0], FLAGS.crop_size[1], 3]
  input_images = tf.placeholder(tf.uint8, tensor_shape, name=_INPUT_NAME)
  input_images = tf.cast(input_images, tf.float32)

  return input_images


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

  with tf.Graph().as_default():
    images = _create_input_tensors()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
        crop_size=FLAGS.crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    predictions, logits = model.predict_labels(
        images,
        model_options=model_options,
        output_softmax=True,
        image_pyramid=FLAGS.image_pyramid)

    semantic_predictions = tf.identity(predictions[common.OUTPUT_TYPE],
                                       name=_OUTPUT_NAME)
    softmax = tf.identity(logits, name=_OUTPUT_SOFTMAX_NAME)

    saver = tf.train.Saver(tf.model_variables())

    tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
    freeze_graph.freeze_graph_with_def_protos(
        tf.get_default_graph().as_graph_def(add_shapes=True),
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        ','.join([_OUTPUT_NAME, _OUTPUT_SOFTMAX_NAME]),
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()
