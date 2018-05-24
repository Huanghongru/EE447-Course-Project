import os
import random
import numpy as np
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import deque

network_dir = "network"

def gen_random_graph_gnp(n, p, uniform_min=0., uniform_max=1.):
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

def gen_random_graph_pow(n, m, p, uniform_min=0., uniform_max=1.):
    """
    Generate a random graph with given nodes n and existence probability p
    All edges are granted a weight represented the probability of being infected
    All nodes are granted a state 0 initially, which means they haven't been infected.

    Parameters:
        n: the number of nodes.
        m: the number of random edges to add for each new node
        p: Probability of adding a triangle after adding a random edge.
        uniform_min: the left point for uniform distribution interval
        uniform_max: the right point for uniform distribution interval
    """
    G = nx.generators.random_graphs.powerlaw_cluster_graph(n, m, p)
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

def visualize(graph):
    pass

def write_graph(graph):
    pass

def load_graph(file):
    """
    Load a large network file into graph variable.
    If the graph doesn't have weight, assign weight randomly.
    
    Parameters:
        file: (string) specify the network file name
    """
    file = os.path.join(network_dir, file)
    G = nx.Graph()

    with open(file, 'r') as f:
        while True:
            datas = f.readlines(10)
            if not datas:
                break
            for data in datas:
                if len(data.split()) > 2:
                    u, v, w = data.split()
                else:
                    u, v = data.split()
                    w = "{0:.2f}".format(random.uniform(0., 1.))
                G.add_edge(eval(u), eval(v), weight=eval(w))
    for _, node_prop in G.nodes(data=True):
        node_prop['state'] = 0
    print "loading graph completed successfully!!"
    print nx.info(G)
    return G


def statistic(graph):
    """
    Print out the basic information of the graph.
    Return weight distribution and degree distribution of the graph.
    """
    print nx.info(graph)
    weights = []
    degrees = []
    for node in graph.nodes():
        degrees.append(graph.degree(node))
    for u, v, d in graph.edges(data=True):
        weights.append(d['weight'])
    return np.array(weights), np.array(degrees)


# def main():
#     G = load_graph('brightkite.txt')
#     w, d = statistic(G)
#     plt.hist(d, bins=1000)
#     plt.show()

# if __name__ == '__main__':
#     main()


