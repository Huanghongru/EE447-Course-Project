import networkx as nx
import numpy as np
import argparse
from utils import *
from algorithm import *

def simulation(graph, q_ratio, hd=True, pagerank=True,
                               ci=True, kcore=True,
                               fanshen=True, average=10):
    """
    Cascade a graph with seeds obtained by different algorithms.
    Parameters:
        graph: a nx.Graph graph
        q_ratio: (float) the ratio of nodes used as seeds.
        average: (int) repeat to average the cascade rate.
    Return:
        a dict of cascade rate with different rate.
    """
    result = {}
    if hd:
        hd_seed = HD(graph.copy(), q_ratio)        
        result['HD_cr'] = 0.
        for i in range(average):
            _, hd_cr = cascade(graph, hd_seed)
            result['HD_cr'] += hd_cr
        result['HD_cr'] /= average
        print "cascade the graph with HD algorithm successfully!!"
    if pagerank:
        pr_seed = PageRank(graph.copy(), q_ratio)
        result['PR_cr'] = 0
        for i in range(average):
            _, pr_cr = cascade(graph, pr_seed)
            result['PR_cr'] += pr_cr
        result['PR_cr'] /= average
        print "cascade the graph with PageRank algorithm successfully!!"
    if ci:
        ci_seed = CI(graph.copy(), q_ratio)
        result['CI_cr'] = 0
        for i in range(average):
            _, ci_cr = cascade(graph, ci_seed)
            result['CI_cr'] += ci_cr
        result['CI_cr'] /= average
        print "cascade the graph with CI algorithm successfully!!"
    if kcore:
        kc_seed = K_core(graph.copy(), q_ratio)
        result['KC_cr'] = 0
        for i in range(average): 
            _, kc_cr = cascade(graph, kc_seed)
            result['KC_cr'] += kc_cr
        result['KC_cr'] /= average
        print "cascade the graph with k-core algorithm successfully!!"
    if fanshen:
        fs_seed = fanshen(graph.copy(), q_ratio)
        _, fs_cr = cascade(graph, fs_seed)
        print "cascade the graph with fanshen algorithm successfully!!"
        result['FS_cr'] = fs_cr
    return result


def simulation_seed(graph, seeds, hd=True, pagerank=True,
                               ci=True, kcore=True,
                               fanshen=True, average=10):
    """
    Cascade a graph with seeds obtained by different algorithms.
    This function is irrelavent to q ratio. Seeds are given.
    Parameters:
        graph: a nx.Graph graph
        seeds: contains seeds from different algorithms
               (hd, pagerank, ci, kcore, fanshen)
        average: (int) repeat to average the cascade rate.
    Return:
        a dict of cascade rate with different rate.
    """
    result = {}
    algo = ['hd', 'pagerank', 'ci', 'kcore', 'fanshen']
    use_check = [hd, pagerank, ci, kcore, fanshen]
    for i in range(len(use_check)):
        if use_check[i]:
            result[algo[i]] = 0
            for j in range(average):
                _, cr = cascade(graph, seeds[i])
                result[algo[i]] += cr
            result[algo[i]] /= average
            print "cascade the graph with {} seeds successfully!!".format(algo[i])
    return result


def real_network_test(graph, sample=True):
    """
    Test algorithm on practical network brightkite.
    Obtain all seed nodes at once to accelerate simulation.

    Parameters:
        graph: (string) name of the practical network.
        sample: (bool) sample before cascading to reduce random effect.
    """
    qratios = np.arange(0.002, 0.14, 0.002)
    # G = load_graph(graph)
    G = nx.read_gexf('random_pow.gexf', node_type=int)
    print nx.info(G)
    print len(G.nodes())*qratios
    algos = [HD, PageRank, K_core, CI, fanshen]
    seeds = []
    for algo in algos:
        seeds.append(algo(G.copy(), max(qratios)))
    print len(seeds[3])

    node_cnt = len(G.nodes())
    for q in qratios:
        N = int(node_cnt*q)
        sub_seeds = [s[:N] for s in seeds]
        if sample:
            G_ = sample_graph(G)
            result = simulation_seed(G_, sub_seeds, fanshen=False, average=1)
        else:
            result = simulation_seed(G, sub_seeds, fanshen=False)
        print "q: {0}\tresult: ".format(q), result

def sample_graph(graph):
    """
    According to the edge weight, we determine a edge whether it can
    spread information before simulation, which may reduce the random
    effect when cascading.

    Parameters:
        graph: a nx.Graph type
    Return:
        a nx.Graph with edge weight ether be 1 or 0.
    """
    G = graph.copy()
    for u, v, d in G.edges(data=True):
        if random.uniform(0, 1) > 1-d['weight']:
            d['weight'] = 1
    return G


def gnp_simulation(n, p, q):
    """
    Run a simulation experiment on a gnp law distribution graph

    Paramters:
        n, p: 2 param to create a gnp graph
        q: (float) fraction of seed    
    """
    G = gen_rand_graph_gnp(n, p)
    print nx.info(G)
    print "result ", simulation(G, q)
    
def pow_simulation(n, m, p, q):
    """
    Run a simulation experiment on a power law distribution graph

    Parameters:
        n, m, p: 3 param to create a power law graph
        q: (float) fraction of seed
    """
    G = gen_random_graph_pow(n, m, p)
    print nx.info(G)
    print "result ", simulation(G, q)

def main():
    # G = gen_random_graph_pow(2000, 2, 0.3)
    # print nx.info(G)
    # nx.write_gexf(G, 'random_pow.gexf')
    # print simulation(G, 0.0015, fanshen=False)
    real_network_test('facebook_combined.txt', sample=True)

if __name__ == '__main__':
    main()