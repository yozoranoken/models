from pathlib import Path

import tensorflow as tf


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def main(unused_argv):
    dataset_path = Path(FLAGS.dataset_dir)
    print(dataset_path)

if __name__ == '__main__':
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
