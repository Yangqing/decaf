"""The main routine that starts a jeffnet demo."""
import cPickle as pickle
from decaf.scripts import jeffnet
from decaf.util import transform
import datetime
import flask
from flask import Flask, url_for, request
import gflags
import logging
import numpy as np
import os
from PIL import Image as PILImage
from skimage import io
import cStringIO as StringIO
import sys
import time
import urllib
from werkzeug import secure_filename

# tornado
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

# local trick
import exifutil

UPLOAD_FOLDER = '/tscratch/tmp/jiayq/decaf'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

gflags.DEFINE_string('net_file', '', 'The network file learned from cudaconv')
gflags.DEFINE_string('meta_file', '', 'The meta file for imagenet.')
gflags.DEFINE_string('bet_file', '', 'The meta file for informative prediction.')
gflags.DEFINE_string('upload_folder', UPLOAD_FOLDER, 'The folder to store the uploaded images.')
FLAGS = gflags.FLAGS

# Obtain the flask app object
app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html',
                                 has_result=False)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    # classify image using the URL
    imageurl = request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = io.imread(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template('index.html',
                                     has_result=True,
                                     result=(False, 'Cannot open image from URL.'))
    logging.info('Image: %s', imageurl)
    result = classify_image(image)
    return flask.render_template('index.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=imageurl)

@app.route('/classify_upload', methods=['POST'])                                     
def classify_upload():
    # classify image using the image name
    try:
        # We will save the file to disk for possible data collection.
        imagefile = request.files['imagefile']
        filename = os.path.join(FLAGS.upload_folder,
                                str(datetime.datetime.now()).replace(' ', '_') + \
                                secure_filename(imagefile.filename))
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
    except Exception as err:
        logging.info('Uploaded mage open error: %s', err)
        return flask.render_template('index.html',
                                     has_result=True,
                                     result=(False, 'Cannot open uploaded image.'))
    result = classify_image(image)
    return flask.render_template('index.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = PILImage.fromarray(image)
    image_pil = image_pil.resize((256,256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

@app.route('/about')
def about():
    return flask.render_template('about.html')

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)

def classify_image(image):
    # let's classify the image.
    try:
        starttime = time.time()
        scores = app.net.classify(image)
        endtime = time.time()
        indices, predictions = app.net.top_k_prediction(scores, 5)
        # In addition to the prediction text, we will also produce the length
        # for the progress bar visualization.
        max_score = scores[indices[0]]
        meta = [(p, '%.5f' % scores[i]) for i, p in zip(indices, predictions)]
        logging.info('result: %s', str(meta))
        # Compute expected information gain
        expected_infogain = np.dot(app.bet['probmat'],
                scores[app.bet['idmapping']]) * app.bet['infogain']
        # sort the scores
        infogain_sort = expected_infogain.argsort()[::-1]
        bet_result = [(app.bet['words'][v], '%.5f' % expected_infogain[v])
                      for v in infogain_sort[:5]]
        logging.info('bet result: %s', str(bet_result))
    except Exception as err:
        logging.info('Classification error: %s', err)
        return (False, 'Oops, something wrong happened wieh classifying the'
                       ' image. Maybe try another one?')
    # If everything is successful, return the results
    return (True, meta, bet_result, '%.3f' % (endtime-starttime))

if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    # try to make the upload directory.
    try:
        os.makedirs(UPLOAD_FOLDER)
    except Exception as err:
        pass
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Loading net...')
    app.net = jeffnet.JeffNet(net_file=FLAGS.net_file,
                              meta_file=FLAGS.meta_file)
    app.bet = pickle.load(open(FLAGS.bet_file))
    # A bias to prefer children nodes in single-chain paths
    # I am setting the value to 0.1 as a quick, simple model - one may want to
    # use better psychological models on the preference of synsets.
    app.bet['infogain'] -= np.array(app.bet['preferences']) * 0.1
    #app.run(host='0.0.0.0')
    logging.info('Starting server...')
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5001)
    IOLoop.instance().start()

