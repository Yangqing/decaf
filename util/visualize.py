"""Implements ways to visualize a decaf net."""
import os
import pydot

LAYER_STYLE = {'style': 'filled', 'fillcolor': '#DFECF3',
              }
BLOB_STYLE = {'style': 'filled', 'fillcolor': 'FEF9E3',
             }

def draw_net(decaf_net, format='png'):
    """Draws a decaf net and returns the image string with the given format.
    The format is defaulted at 'png'.
    """
    pydot_graph = pydot.Dot(graph_type='digraph')
    pydot_nodes = {}
    for name in decaf_net.layers:
        pydot_nodes[name] = pydot.Node(name, **LAYER_STYLE)
    for name in decaf_net.blobs:
        pydot_nodes[name] = pydot.Node(name, **BLOB_STYLE)
    for name in pydot_nodes:
        pydot_graph.add_node(pydot_nodes[name])
    for parent, child in decaf_net.graph.edges():
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[parent], pydot_nodes[child]))
    return pydot_graph.create(format=format)

def draw_net_to_file(decaf_net, filename):
    """Draws a decaf net, and saves it to file using the format given as the
    file extension.
    """
    format = os.path.splitext(filename)[-1]
    with open(filename, 'w') as fid:
        fid.write(draw_net(decaf_net, format))
