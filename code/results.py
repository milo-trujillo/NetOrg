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
        self.welfare = welfare
        self.num_agents = num_agents
        self.num_env = num_env

    def reset(self):
        self.get_trimmed_listen_params()

    def get_trimmed_listen_params(self, cutoff=.1):
        self.trimmed = []
        for lparams in self.listen_params:
            maxp = np.max(lparams)
            lparams = lparams * np.int_(lparams * lparams>(cutoff*maxp))
            self.trimmed.append(lparams)

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
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
