"""The main routine that starts a jeffnet demo."""
from decaf.scripts import jeffnet
import flask
from flask import Flask, url_for, request
import logging
from skimage import io
import StringIO
import urllib
import time

# Obtain the flask app object
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    imageurl = request.args.get('imageurl', '')
    if imageurl == '':
        # Simply render the default page.
        has_result = False
        result = []
    else:
        has_result = True
        # get the results.
        result = classify_image_url(imageurl)
    return flask.render_template('index.html',
                                 has_result=has_result,
                                 result=result,
                                 imageurl=imageurl)

@app.route('/about')
def about():
    return 'Under Construction.'

def classify_image_url(imageurl):
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
        image = io.imread(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('Image open error: %s', err)
        return (False, 'Cannot open image.')
    # let's classify the image.
    try:
        starttime = time.time()
        scores = app.net.classify(image)
        indices, predictions = app.net.top_k_prediction(scores, 5)
        # In addition to the prediction text, we will also produce the length
        # for the progress bar visualization.
        max_score = scores[indices[0]]
        meta = [(p, '%.5f' % scores[i]) for i, p in zip(indices, predictions)]
        logging.info('Image: %s', imageurl)
        logging.info('result: %s', str(meta))
    except Exception as err:
        logging.info('Classification error: %s', err)
        return (False, 'Error classifying the image.'
                       ' Please send the url to Yangqing!')
    # If everything is successful, return the results
    endtime = time.time()
    return (True, meta, str(endtime-starttime))
        
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    app.net = jeffnet.JeffNet()
    app.run(host='0.0.0.0')
