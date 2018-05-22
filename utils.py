import os
import random
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
    Using multiprocessing to accelerate this process.
    """
    def read_edge(graph, data):
        data_ = data.split()
        if len(data_) > 2:
            u, v, w = data_
        else:
            u, v = data_
            w = 0
        graph.add_edge(u, v, weight=w)

    def process_wrapper(graph, file, lineByte):
        with open(file) as f:
            f.seek(lineByte)
            line = f.readline()
            read_edge(graph, line)

    file = os.path.join(network_dir, file)
    G = nx.Graph()

    pool = mp.Pool(4)
    jobs = []

    with open(file) as f:
        nextLineByte = f.tell()
        for line in f:
            jobs.append(pool.apply_async(process_wrapper, (G, file, nextLineByte)))
            nextLineByte = f.tell()

    for job in jobs:
        job.get()

    pool.close() 
    return G


def statistic(graph):
    #TODO: return some important properties of the given graph.
    #TODO: maybe include number of nodes, edges, degree and average weight, etc.
    pass




