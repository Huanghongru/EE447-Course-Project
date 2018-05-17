import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


def gen_random_graph(n, p, uniform_min=0., uniform_max=1.):
    """
    Generate a random graph with given nodes n and existence probability p
    All edges are granted a weight represented the probability of being infected
    All nodes are granted a state 0 initially, which means they haven't been infected.

    Parameters:
        n: The number of nodes.
        p: Probability for edge creation.
        uniform_min: the left point for uniform distribution interval
        uniform_max: the right point for uniform distribution interval
    """
    if p < 0.01:
        G = nx.generators.random_graphs.fast_gnp_random_graph(n, p)
    else:
        G = nx.generators.random_graphs.gnp_random_graph(n, p)

    for _, node_prop in G.nodes(data=True):
        node_prop['state'] = 0
    for _, _, edge_prop in G.edges(data=True):
        edge_prop['weight'] = float("{0:.2f}".format(random.uniform(uniform_min, uniform_max)))
    return G

def print_graph(graph, node=True, edge=True):
    """
    Print out all information of the given graph
    """
    if node:
        for _, node_prop in graph.nodes(data=True):
            print _, node_prop
    if edge:
        for u, v, edge_prop in graph.edges(data=True):
            print "({},{})\t{}".format(u, v, edge_prop)
