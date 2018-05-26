import networkx as nx
import numpy as np
import time
import sys
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
    Return the N*q_ratio nodes with the highest degree.
    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes
    """
    N = int(len(graph.nodes())*q_ratio)
    # There is a very annoying bug here about the variable type returned 
    # by Graph.degree() method. When you upload it to server to run, 
    # change sorted part as:
    #
    #   tmp_dlist = sorted(degree_list, key=lambda x: degree_list[x], reverse=True)
    #
    # to avoid this bug. 
    degree_list = graph.degree() 
    tmp_dlist = sorted(degree_list, key=lambda nd_pair: nd_pair[1], reverse=True)
    return [nd_pair[0] for nd_pair in tmp_dlist[:N]]

def HDA(graph, q_ratio):
    """
    Return the N*q_ratio nodes with the highest degree.
    Whenever remove a node from the graph, recalculate the degrees.
    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes   
    """
    N = int(len(graph.nodes())*q_ratio)
    seed = []
    while len(seed) < N:
        degree_list = graph.degree()
        tmp_dlist = sorted(degree_list, key=lambda nd_pair: nd_pair[1], reverse=True)
        seed.append(tmp_dlist[0][0])
        graph.remove_node(tmp_dlist[0][0])
    return seed

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
            # There is a very annoying bug here about the variable type returned 
            # by Graph.degree() method. When you upload it to server to run, 
            # change sorted part as:
            #
            #   tmp_dlist = sorted(degree_list, key=lambda x: degree_list[x], reverse=True)
            #
            # change the for loop part as:
            #
            #   for node in tmp_dlist:
            #       degree = graph.degree(node)
            #       ...
            #       ...
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

        Return:
            paths: 
        """
        ball_nodes = set()
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
                    if cur_path[-1] not in ball_nodes:
                        paths.append(cur_path)
                        for node_ in cur_path:
                            affected[node_].add(node)
                    ball_nodes.add(cur_node)

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
        if not node_status[node]:
            return 0
        ci = 0
        for path in paths:
            if not node_status[path[-1]]:
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

    # a list of nodes that will be affected
    #         for each node if it is removed.
    affected = [set() for i in range(len(graph.nodes()))]

    node_count = 0
    path_count = 0
    for node in graph.nodes():
        node_path = ball(node, l)
        path_count += len(node_path)
        all_paths.append(node_path)
        node_count += 1
        if node_count % 1000 == 0:
            print "{} nodes have been processed...".format(node_count)

    print "ball algorithm completed!!"
    print "paths count: ", path_count

    # record each node's CI value, only update nodes that are influenced
    # by the removal.
    CIs = [0 for i in range(node_count)]
    to_be_update = graph_.nodes()
    while len(seed) < N:
        for node in to_be_update:
            CIs[node] = get_CI(node, all_paths[node], node_status)
        max_CI_node, max_CI = max(enumerate(CIs), key=lambda ci: ci[1])
        if max_CI == 0:
            seed.extend(list(graph_.nodes())[:N-len(seed)])
            break
        seed.append(max_CI_node)
        node_status[max_CI_node] = 0
        CIs[max_CI_node] = 0

        # The CI value of a node is not abled to increase.
        # so a node with a CI value of 0 can be eliminated from update list.
        to_be_update = []
        for n in affected[max_CI_node]:
            if CIs[n]:
                to_be_update.append(n)

        # The max CI value may be zero after some removal
        if max_CI_node in graph_.nodes():
            graph_.remove_node(max_CI_node)
        print "appending seed node {}/{}".format(len(seed), N)
    return seed

def fanshen(graph, q_ratio, l=3):
    """
    The main theory is the same as CI's, whereas the detail CI calculations differ.

    Parameters:
        graph: a nx.Graph type.
        q_ratio: (float) the ratio of nodes used as seeds.
    Return:
        a list of seed nodes
    """

    def ball(node, l):
        """
        Same as CI's inner function.

        Return:
            paths: 
        """
        ball_nodes = set()
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
                    if cur_path[-1] not in ball_nodes:
                        paths.append(cur_path)
                        # for node_ in cur_path:
                            # affected[node_].add(node)
                    ball_nodes.add(cur_node)

                for u, v in graph.edges(cur_node):
                    if v not in visited:
                        aug_path = cur_path[:]
                        aug_path.append(v)
                        q.append((v, aug_path))
                cur_layer_nodes_num -= 1
            cur_len += 1

        return paths

    def get_u2(i, j, node_status):
        if node_status[i] == 1:
            return 1.

        neighbors = list(graph[i])
        neighbors.remove(j)
        # print i,neighbors
        prod = 1.
        # if neighbors != None:
        for k in neighbors:
            p_ki = graph[k][i]['weight']
            u_ki = u_dic[(k,i)]
            prod *= 1. - p_ki * u_ki
        u_ij = 1. - prod
        return u_ij

    def get_CI(node, paths, node_status):
        """
        Different calculations.
        """

        def get_u1(i, j):
            if node_status[i] == 1:
                return 1.

            prod = 1. - u_ij
            p_ji = graph[j][i]['weight']
            u_ji = u_dic[(j,i)]
            prod *= 1. - p_ji * u_ji
            u_i = 1. - prod
            return u_i

        if node_status[node] or graph.degree(node) == 1:
            return 0.

        ci = 0.
        for path in paths:
            B_ij = np.mat([1,-1])
            for start in range(len(path)-1):
                i = path[start]
                j = path[start+1]
                # First calculate u_ij:
                # the probability that i get infected in G\j
                u_ij = u_dic[(i,j)]
                u_i = get_u1(i, j)
                p_ij = graph[i][j]['weight']

                b00 = 1
                b01 = u_ij - 1
                b10 = (1 - p_ij) / (1 - p_ij * u_ij)
                b11 = (1 - p_ij - u_i + p_ij * u_ij) / (p_ij * u_ij - 1)
                B_ij = B_ij * np.mat([[b00, b01], [b10, b11]])
            B_ij = B_ij * np.mat([[1],[-1]])
            ci += float(B_ij)
        ci *= graph.degree(node) - 1
        return ci


    N = int(len(graph.nodes())*q_ratio)
    seed = []
    node_status = [0 for i in range(len(graph.nodes()))]
    all_paths = []
    u_dic = dict()
    for edge in graph.edges():
        u_dic[edge] = 0.
        u_dic[edge[::-1]] = 0.

    node_count = 0
    path_count = 0
    for node in graph.nodes():
        node_path = ball(node, l)
        path_count += len(node_path)
        all_paths.append(node_path)
        node_count += 1
        if node_count % 1000 == 0:
            print "{} nodes have been processed...".format(node_count)

    print "ball algorithm completed!!"
    print "paths count: ", path_count

    CIs = [0 for i in range(node_count)]
    to_select = list(graph.nodes())
    to_update = set(graph.nodes())
    while True:
        for node in to_update:
            CIs[node] = get_CI(node, all_paths[node], node_status)
        max_CI_node, max_CI = max(enumerate(CIs), key=lambda ci: ci[1])
        print "max_CI_node:{},max_CI:{}".format(max_CI_node, max_CI)
        if max_CI == 0:
            seed.extend(to_select[:N-len(seed)])
            break

        seed.append(max_CI_node)
        if len(seed) == N:
            break
        node_status[max_CI_node] = 1
        CIs[max_CI_node] = 0
        to_select.remove(max_CI_node)

        # here consider the case of multiple connected components
        # whereas generally the graph is connectd
        # so here may offer some space for optimizing
        to_update = nx.node_connected_component(graph, max_CI_node)
        to_update.remove(max_CI_node)
        for edge in graph.edges(max_CI_node):
            u_dic[edge] = 1.
        
        '''Iterating process'''
        # Python2 does not support nonlocal definition for u_dic
        # so i cannot use a function

        epoch = 0
        while True:
            epoch += 1
            max_bias = 0
            tmp_bias = 0
            for u in to_update:
                for i, j in graph.edges(u):
                    u_ij = get_u2(i, j, node_status)
                    tmp_bias = abs(u_ij - u_dic[(i,j)])
                    # if tmp_bias > 0:
                    # print i,j,u_ij,u_dic[(i,j)]
                    # exit()
                    max_bias = max(tmp_bias, max_bias)
                    u_dic[(i,j)] = u_ij
            print "Iterating epoch {}: max bias is {}".format(epoch, max_bias)
            if max_bias < 0.01:
                break
        '''end iterating'''

        print "appending seed node {}/{}".format(len(seed), N)

    return seed


def main():
    # G = gen_random_graph_pow(1000, 1, 0.5)
    # G = gen_random_graph_gnp(1000, 0.001)
    # nx.write_gexf(G, 'random_pow.gexf')
    # nx.write_gexf(G, 'random_gnp.gexf')
    # print nx.info(G)
    # G = nx.read_gexf('random_pow.gexf', node_type=int)
    G = nx.read_gexf('random_gnp.gexf', node_type=int)

    print nx.info(G)
    t1 = time.time()
    # print CI(G, 0.0015)
    for i in G.edges(450):
        print i

    print fanshen(G, 0.015)
    print CI(G, 0.015)
    print time.time()-t1

if __name__ == '__main__':
     main() 