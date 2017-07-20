#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # Don't require a GUI to generate graphs
from matplotlib import pyplot as plt
import networkx as nx
plt.ion()

class Results(object):
    def __init__(self, training_res, listen_params, num_agents, num_env, welfare):
        self.training_res = training_res
        self.listen_params = listen_params
        self.get_trimmed_listen_params()
        self.get_meta_agent_weights()
        self.welfare = welfare
        self.num_agents = num_agents
        self.num_env = num_env

    def reset(self):
        self.get_trimmed_listen_params()
        self.get_meta_agent_weights()

    def get_trimmed_listen_params(self, cutoff=.5):
        self.trimmed = []
        for lparams in self.listen_params:
            maxp = np.max(lparams)
            lparams = lparams * np.int_(lparams * lparams>(cutoff*maxp))
            self.trimmed.append(lparams)

    def get_meta_agent_weights(self):
        self.meta_weights = []
        for i in range(self.num_agents):
            self.meta_weights.append(self.get_agent_weights(i))

    # Concats the listen weights from each layer of an agent
    def get_agent_weights(self, agent):
        weights = []
        total_nodes = len(self.trimmed)
        while( agent < total_nodes ):
            weights += self.trimmed[agent].tolist()[0]
            agent += self.num_agents
        return weights

    def generate_graph(self, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
        G = nx.DiGraph()
        vpos = 0
        hpos = 0
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), pos=(hpos, vpos))
            hpos += hspace
        vpos = vspace
        hpos = 0
        hspace = hspace*numenv/float(len(self.trimmed))
        highest_listened = numenv - 1
        for aix, agent in enumerate(self.trimmed):
            hpos += hspace
            nextlevel = False
            nodenum = numenv +aix
            G.add_node(nodenum, color='r', name = "A" + str(aix))
            for eix, val in enumerate(agent.flatten()):
                if abs(val) > 0:
                    G.add_edge(eix, nodenum, width=val)
                    if eix > highest_listened:
                        highest_listened =eix
                        nextlevel=True
            if nextlevel:
                vpos += vspace
            G.node[nodenum]["pos"] = (hpos, vpos)
        return G

    def graph_org(self, vspace=1, hspace=2):
        G = self.generate_graph(vspace, hspace)
        colors = nx.get_node_attributes(G, "color").values()
        pos= nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, node_color = colors, with_labels=True,
                    labels=nx.get_node_attributes(G, "name"), alpha=.5, node_size=600 )
        return G

    def graph_cytoscape(self, filename, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        G = nx.DiGraph()
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), category="environment")
        for aix, agent in enumerate(self.trimmed):
            nodenum = int(numenv + aix)
            prefix = aix % self.num_agents
            layer = aix / self.num_agents
            n = "A%d_%d" % (prefix, layer)
            G.add_node(nodenum, color='r', name=n, category="agent", layer=layer)
            # For each node, weights will be zero if the edge should be ignored
            # and otherwise represent the cost of the edge
            for dest, weight in enumerate(agent.flatten()):
                # We need to offset the other end of the arrow, because agents
                # after layer0 don't listen to env, and only listen to the layer
                # immediately below them
                if( layer > 0 ):
                    dest += self.num_env
                    dest += (self.num_agents * (layer-1))
                if( abs(weight) > 0 ):
                    G.add_edge(int(dest), nodenum, width=float(weight),
                        weight=float(abs(weight)))
            if( layer > 0 ):
                predecessor = int(numenv + aix - self.num_agents)
                G.add_edge(predecessor, nodenum, width=0, weight=0)
        nx.write_graphml(G, filename)
        #nx.write_gml(G, filename)

    # Returns a list of the *agents* a given agent is connected to
    def get_non_zero_edges_to_agents(self, agent):
        neighbors = []
        weights = self.meta_weights[agent]
        for i in range(len(weights)):
            if( i >= self.num_env and weights[i] != 0 ):
                neighbors.append((i - self.num_env) % self.num_agents)
        return neighbors

    # Returns sum of all weights along the path, or zero if path impossible
    def get_weight_of_path(self, src, dst):
        path = self.get_path(src, dst)

    # Performs a simple breadth-first search to return path between agents
    # Iterative, so we won't hit max stack depth on a big graph
    def get_path(self, src, dst):
        total_weight = 0.0
        seen = set([src])
        parents = dict()
        q = self.get_non_zero_edges_to_agents(src)
        while( len(q) > 0 ):
            print "Q is now " + str(q)
            c = q.pop(0)
            seen.add(c)
            if( c == dst ):
                path = [c]
                while( c in parents.keys() ):
                    c = parents[c]
                    path.append(c)
                path.reverse()
                return path
            for n in self.get_non_zero_edges_to_agents(c):
                if( n not in seen ):
                    seen.add(n)
                    parents[n] = c
                    q.append(n)
        return []
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
