import networkx as nx
import numpy as np
from utils import *

def cascade(graph, seed, verbose=False):
    """
    Cascade the graph with given seed nodes. Essentially it is a graph traversal.
    Parameters:
        graph: a networkx graph type
        seed: a list of seed node id
        verbose: print detailed info while cascading, default to False
    Return:
        the graph after cascading. (In fact the graph variable outside the function will also be changed...)
        infected rate after cascading.
    """
    def casc_rate(graph):
        infected = 0
        for node, attrib in graph.nodes(data=True):
            infected += attrib['state']
        return float(infected) / len(graph.nodes())

    for s in seed:
        graph.node[s]['state'] = 1
    q = deque(seed)
    while len(q):
        if not verbose:
            cur_node = q.pop()
            for u, v, d in graph.edges(cur_node, data=True):
                if not graph.node[v]['state']:
                    if random.uniform(0, 1) > 1-d['weight']:
                        graph.node[v]['state'] = 1
                        q.append(v)
        else:
            cur_node = q.pop()
            for u, v, d in graph.edges(cur_node, data=True):
                if not graph.node[v]['state']:
                    casc_prop = random.uniform(0, 1)
                    print "{}->{}\t\tweight:{}\tcasc prop:{:.2f}".format(u, v, d['weight'], casc_prop),
                    if casc_prop > 1-d['weight']:
                        print "\tcascade success XD"
                        graph.node[v]['state'] = 1
                        q.append(v)
                    else:
                        print "\tcascade faile :("
            print "="*60
            print "after node {} cascade its information, casc_rate:{}".format(u, casc_rate(graph))
            print "="*60

    return graph, casc_rate(graph)

def HD(graph, q_ratio):
    """
    Return the N*q_ratio nodes with the highest degrees.
    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes
    """
    N = int(len(graph.nodes())*q_ratio)
    degree_list = graph.degree()
    tmp_dlist = sorted(degree_list, key=lambda nd_pair: nd_pair[1], reverse=True)
    return [nd_pair[0] for nd_pair in tmp_dlist[:N]]

def PageRank(graph, q_ratio):
    """
    """
    print "hello world"

def K_core(graph, q_ratio):
    """
    The k-core is the largest subgraph where vertices have at least k interconnections.
    In this project, we need a set of seed nodes with size of N. So we iteratively increase
    k and remove all the vertices with degree less than k. And we randomly select N nodes
    from the final subgraph.

    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes  
    """
    k = 1
    N = int(len(graph.nodes())*q_ratio)
    while len(graph.nodes()) > N:
        to_be_remove = []
        d_list = list(graph.degree())

        # shuffle the degree list in case always selecting the nodes id from small to large
        random.shuffle(d_list)  

        for node, degree in d_list:
            if degree <= k and len(graph.nodes())-len(to_be_remove) > N:
                to_be_remove.append(node)
        graph.remove_nodes_from(to_be_remove)
        k += 1 
    return list(graph.nodes())[:N]


def CI(graph, q_ratio):
    """
    """
    pass

def fanshen(graph, q_ratio):
    """
    """
    pass

def main():
    G = gen_random_graph(30, 0.1, uniform_min=0.5, uniform_max=0.5)
    print G.degree()
    print K_core(G.copy(), 0.1)
    print HD(G.copy(), 0.1)

if __name__ == '__main__':
    main()
