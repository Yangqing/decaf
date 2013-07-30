"""Implements ways to visualize a decaf net."""
import os
import pydot

LAYER_STYLE = {'shape': 'record', 'fillcolor': '#DFECF3',
               'style': 'filled'}
BLOB_STYLE = {'shape': 'record', 'fillcolor': '#FEF9E3',
              'style': 'filled'}

def draw_net(decaf_net, format='png'):
    """Draws a decaf net and returns the image string with the given format.
    The format is defaulted at 'png'.
    """
    pydot_graph = pydot.Dot(graph_type='digraph')
    pydot_nodes = {}
    for name, layer in decaf_net.layers.iteritems():
        pydot_nodes[name] = pydot.Node(
            '%s|<%s>' % (name, layer.__class__.__name__), **LAYER_STYLE)
    for name, blob in decaf_net.blobs.iteritems():
        if blob.has_data():
            shapestr = 'x'.join(str(v) for v in blob.data().shape[1:])
        else:
            shapestr = 'unknown shape'
        pydot_nodes[name] = pydot.Node(
            '%s|<%s>' % (name, shapestr),
            **BLOB_STYLE)
    for name in pydot_nodes:
        pydot_graph.add_node(pydot_nodes[name])
    print decaf_net.graph.edges()
    for parent, child in decaf_net.graph.edges():
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[parent], pydot_nodes[child]))
    return pydot_graph.create(format=format)

def draw_net_to_file(decaf_net, filename):
    """Draws a decaf net, and saves it to file using the format given as the
    file extension.
    """
    format = os.path.splitext(filename)[-1][1:]
    with open(filename, 'w') as fid:
        fid.write(draw_net(decaf_net, format))
