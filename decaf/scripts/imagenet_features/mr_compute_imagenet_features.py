from mincepie import mapreducer, launcher
import gflags
import glob
import numpy as np
import os
from skimage import io
import uuid
import logging
from decaf.scripts.jeffnet import JeffNet

# constant value
FEATURE_DTYPE = np.float32

gflags.DEFINE_string("input_folder", "",
                     "The folder that contains all input tar files")
gflags.DEFINE_string("output_folder", "",
                     "The folder that we write output features to")
gflags.DEFINE_boolean("single", False,
                      "If True, the input is single image names.")
gflags.DEFINE_string("feature_name", "fc6_cudanet_out",
                     "The feature name.")
gflags.DEFINE_integer("randprojection", 0,
                      "If positive, perform random projection down to the "
                      " given dimension.")
FLAGS = gflags.FLAGS


class DecafnetMapper(mapreducer.BasicMapper):
    """The ImageNet Compute mapper. The input value would be a synset name.
    """
    def set_up(self):
        logging.info('Loading jeffnet...')
        # We will simply instantiate a jeffnet manually
        data_root = '/u/vis/common/deeplearning/models/'
        self._net = JeffNet(data_root+'imagenet.jeffnet.epoch90',
                data_root+'imagenet.jeffnet.meta')
        logging.info('Jeffnet loaded.')
        self._randproj = None

    def map(self, key, value):
        if type(value) is not str:
            value = str(value)
        if FLAGS.single:
            files = [os.path.join(FLAGS.input_folder, value)]
        else:
            files = glob.glob(os.path.join(FLAGS.input_folder, value, '*.JPEG'))
        files.sort()
        logging.info('%s: %d files', value, len(files))
        features = None

        for i, f in enumerate(files):
            try:
                img = io.imread(f)
                logging.info(f)
                _ = self._net.classify(img)
                feat = self._net.feature(FLAGS.feature_name) 
                dim = np.prod(feat.shape[1:])
                if FLAGS.randprojection > 0:
                    if self._randproj is None:
                        np.random.seed(1701)
                        self._randproj = np.random.randn(dim, FLAGS.randprojection).astype(FEATURE_DTYPE)
                        self._randproj /= dim
                    feat = np.dot(feat.reshape(feat.shape[0], dim), self._randproj)
                if features is None:
                    features = np.zeros((len(files) * 10,) + feat.shape[1:],
                                        dtype = FEATURE_DTYPE)
                features[i*10:(i+1)*10] = feat
            except IOError:
                # we ignore the exception (maybe the image is corrupted or
                # pygist has some bugs?)
                print f, Exception, e
        outname = str(uuid.uuid4()) + '.npy'
        try:
            os.makedirs(FLAGS.output_folder)
        except OSError:
            pass
        np.save(os.path.join(FLAGS.output_folder, outname), features)
        yield value, outname

mapreducer.REGISTER_DEFAULT_MAPPER(DecafnetMapper)

class DecafnetReducer(mapreducer.BasicReducer):
    def reduce(self, key, values):
        """The Reducer basically renames the numpy file to the synset name
        Input:
            key: the synset name
            value: the temporary name from map
        """
        os.rename(os.path.join(FLAGS.output_folder, values[0]),
                os.path.join(FLAGS.output_folder, key + '.npy'))
        return key

mapreducer.REGISTER_DEFAULT_REDUCER(DecafnetReducer)
mapreducer.REGISTER_DEFAULT_READER(mapreducer.FileReader)
mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.FileWriter)

if __name__ == "__main__":
    launcher.launch()
