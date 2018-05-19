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
            cur_node = q.popleft()  # FIFO
            for u, v, d in graph.edges(cur_node, data=True):
                if not graph.node[v]['state']:
                    if random.uniform(0, 1) > 1-d['weight']:
                        graph.node[v]['state'] = 1
                        q.append(v)
        else:
            cur_node = q.popleft()
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
    PageRank outputs a probability distribution used to represent the likelihood 
    that a person randomly clicking on links will arrive at any particular page. 
    The higher the probability, the higher the PR value of this page. 
    Here we consider each node as a page, using PageRank() provided by networkx.

    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes
    """
    N = int(len(graph.nodes())*q_ratio)
    pr = nx.pagerank(graph)
    descend_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in descend_pr[:N]]


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
    def k_decomposition(k, N):
        """
        Obtain the subgraph after k-core algorithm
        """
        while True:
            degree_list = graph.degree()
            tmp_dlist = sorted(degree_list, key=lambda nd_pair: nd_pair[1], reverse=True)
            to_be_remove = []
            for node, degree in tmp_dlist:
                if degree <= k and len(graph.nodes())-len(to_be_remove) > N:
                    to_be_remove.append(node)
            if not to_be_remove:
                break
            else:
                graph.remove_nodes_from(to_be_remove)


    k = 1
    N = int(len(graph.nodes())*q_ratio)
    while len(graph.nodes()) > N:
        k_decomposition(k, N)
        k += 1

    # shuffle the degree list in case always selecting the nodes id from small to large
    seed_candidate = list(graph.nodes())
    random.shuffle(seed_candidate)
    return seed_candidate[:N]

# TODO: some bugs may exist so CI peform worse than K-core
def CI(graph, q_ratio, l=3):
    """
    Looking for the node with the largest CI value. Virtually remove
    the nodes and find the next node with the largest CI value.
    Repeat until we get N seed nodes.

    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes   
    """ 
    def ball(node, l):
        """
        Given a node i, return all the nodes that are distance l from i
        and all the corresponding paths.

        Note that we use the original graph for ball detection.

        The last node in the path is the ball node.
        """
        # ball_nodes = set()
        paths = []
        visited = set()
        cur_len = 0
        min_dist = {}   # record the minimum distance from node i
        q = deque([(node, [node])])

        while cur_len <= l:
            cur_layer_nodes_num = len(q)
            while cur_layer_nodes_num > 0 and len(q):
                cur_node, cur_path = q.popleft()
                visited.add(cur_node)

                if not min_dist.get(cur_node):
                    min_dist[cur_node] = cur_len
                
                if cur_len == l and min_dist[cur_node] == l:
                    # ball_nodes.add(cur_node)
                    paths.append(cur_path)

                for u, v in graph.edges(cur_node):
                    if v not in visited:
                        aug_path = cur_path[:]
                        aug_path.append(v)
                        q.append((v, aug_path))
                cur_layer_nodes_num -= 1
            cur_len += 1

        return paths

    def get_CI(node, paths, node_status):
        """
        Calculate the CI value according to the equation.
        Note that we use the copy graph to calculate CI. The degrees
        are different from the original graph.
        """
        ci = 0
        for path in paths:
            if path[-1] not in graph_.node():
                continue

            nk = 1
            for p_node in path:
                nk *= node_status[p_node]
            nk *= (graph_.degree(path[-1])-1)
            ci += nk
        return ci * (graph_.degree(node)-1)


    N = int(len(graph.nodes())*q_ratio)

    seed = []
    graph_ = graph.copy()   # copy a graph for node removal 
    node_status = [1 for i in range(len(graph.nodes()))]
    all_paths = []
    for node in graph.nodes():
        all_paths.append(ball(node, l))

    while len(seed) < N:
        max_CI, max_id = 0, 0
        for node in graph_.nodes():
            cur_CI = get_CI(node, all_paths[node], node_status)
            if cur_CI > max_CI:
                max_CI = cur_CI
                max_id = node

        graph_.remove_node(max_id)
        node_status[max_id] = 0
        seed.append(max_id)
    return seed

def fanshen(graph, q_ratio):
    """
    """
    pass


