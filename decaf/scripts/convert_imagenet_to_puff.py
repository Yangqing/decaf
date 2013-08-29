"""This script converts the imagenet images to the puff format, using the
standard protocol adopted by Alex Krizhevsky in his ILSVRC2012 winner
strategy. Specifically, each image is scaled so that the shorter edge has
256 pixels, and then the center part is cropped to produce the final image.

In addition to the conversion above, we also do the following two things:
    1. Create labels from folder names. Instead of random ordering, we will
       sort the labels in their ASCII order to create the label values,
       starting from 0.
    2. Randomly shuffle data so that images of different labels do not come
       sequentially. This may make classifier training more reasonable (by
       having a pseudo-random order) in the sense that each minibatch may
       contain images from multiple classes.

Usage:
    convert_imagenet_to_puff.py
        --root=/path/to/imagenet/images
        --output=/name/of/the/output/feature/file
        --output_label=/name/of/the/output/label/file
        [--size=256]
        [--extension=JPEG]
where the path to imagenet should be a folder that has different synsets as
its subfolders, which contain JPEG images.
"""

from decaf import puff
from decaf.util import Timer
import gflags
import glob
import logging
import numpy as np
import random
from skimage import io, transform
import os
import sys

# Flags
gflags.DEFINE_string('root', '', 'The path to the imagenet data.')
gflags.DEFINE_string('output', '', 'The output file name.')
gflags.DEFINE_string('output_label', '', 'The output label file name.')
gflags.DEFINE_integer('size', 256, 'The image size. If size=0,'
                      ' no resize is carried out.')
gflags.DEFINE_string('extension', 'JPEG', 'The image extension.')
gflags.MarkFlagAsRequired('root')
gflags.MarkFlagAsRequired('output')
gflags.MarkFlagAsRequired('output_label')
FLAGS = gflags.FLAGS

def process_image(filename):
    img = io.imread(filename)
    if img.ndim == 2:
        img = np.tile(img[:,:,np.newaxis], (1,1,3))
    if FLAGS.size == 0:
        return img
    if img.shape[0] < img.shape[1]:
        newshape = (FLAGS.size,
                    int(img.shape[1] * float(FLAGS.size) / img.shape[0] + 0.5))
    else:
        newshape = (int(img.shape[0] * float(FLAGS.size) / img.shape[1] + 0.5),
                    FLAGS.size)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    img = transform.resize(img, newshape)
    # now, cut the margin
    if img.shape[0] > FLAGS.size:
        offset = int(img.shape[0] - FLAGS.size)
        return (img[offset:offset+FLAGS.size] * 255).astype(np.uint8)
    else:
        offset = int(img.shape[1] - FLAGS.size)
        return (img[:, offset:offset+FLAGS.size] * 255).astype(np.uint8)


def main(argv):
    logging.getLogger().setLevel(logging.INFO)
    FLAGS(argv)
    logging.info('Listing images...')
    images = glob.glob(os.path.join(FLAGS.root, '*', '*.' + FLAGS.extension))
    random.shuffle(images)
    logging.info('A total of %d images', len(images))
    labels = list(set(os.path.dirname(image) for image in images))
    labels.sort()
    logging.info('A total of %d identical classes', len(labels))
    label2id = dict((n, i) for i, n in enumerate(labels))
    # Now, process individual images, and write them to the puff files.
    image_writer = puff.PuffStreamedWriter(FLAGS.output)
    labels = []
    current = 0
    my_timer = Timer()
    for filename in images:
        image_cropped = process_image(filename)
        labels.append(label2id[os.path.dirname(filename)])
        image_writer.write_single(image_cropped)
        current += 1
        if current % 1000 == 0:
            logging.info('Processed %d images, elapsed %s',
                         current, my_timer.lap())
    image_writer.finish()
    # Write the label
    puff.write_puff(np.array(labels, dtype=np.int), FLAGS.output_label)
    logging.info('Done.')


if __name__ == '__main__':
    main(sys.argv)
