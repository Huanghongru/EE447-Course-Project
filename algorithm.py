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

    return graph



def main():
    G = gen_random_graph(20, 0.4, uniform_min=0.5, uniform_max=0.5)
    # cascade(G, [3], verbose=True)
    print_graph(G)
    # plt.subplot(121)
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()

if __name__ == '__main__':
    main()