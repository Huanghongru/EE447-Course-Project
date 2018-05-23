import networkx as nx
import numpy as np
import argparse
from utils import *
from algorithm import *

def simulation(graph, q_ratio, hd=True, pagerank=True,
                               ci=True, kcore=True,
                               fanshen=True):
    """
    Cascade a graph with seeds obtained by different algorithms.
    Parameters:
        graph: a nx.Graph graph
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a dict of cascade rate with different rate.
    """
    result = {}
    if hd:
        hd_seed = HD(graph.copy(), q_ratio)
        _, hd_cr = cascade(graph, hd_seed)
        print "cascade the graph with HD algorithm successfully!!"
        result['HD_cr'] = hd_cr
    if pagerank:
        pr_seed = PageRank(graph.copy(), q_ratio)
        _, pr_cr = cascade(graph, pr_seed)
        print "cascade the graph with PageRank algorithm successfully!!"
        result['PR_cr'] = pr_cr
    if ci:
        ci_seed = CI(graph.copy(), q_ratio)
        _, ci_cr = cascade(graph, ci_seed)
        print "cascade the graph with CI algorithm successfully!!"
        result['CI_cr'] = ci_cr
    if kcore:
        kc_seed = K_core(graph.copy(), q_ratio)
        _, kc_cr = cascade(graph, kc_seed)
        print "cascade the graph with k-core algorithm successfully!!"
        result['KC_cr'] = kc_cr
    if fanshen:
        fs_seed = fanshen(graph.copy(), q_ratio)
        _, fs_cr = cascade(graph, fs_seed)
        print "cascade the graph with fanshen algorithm successfully!!"
        result['FS_cr'] = fs_cr
    return result

def cascRate_vs_qRatio(graph, q_ratio_group):
    """
    """
    pass

def main():
    G = gen_random_graph_gnp(2000, 0.0015, uniform_max=0.5)
    # print simulation(G, 0.005, fanshen=False)
    # G = load_graph('brightkite.txt')
    print simulation(G, 0.005, fanshen=False)

if __name__ == '__main__':
    main()