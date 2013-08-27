import cPickle as pickle
from decaf.util import translator 
from decaf.util import visualize

layers = pickle.load(open('cifar_11pct_layers.alexnet'))                                       
decaf_net = translator.translate_cuda_network(layers)
visualize.draw_net_to_file(decaf_net, '/u/jiayq/.public_html/translator.png')
