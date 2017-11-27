#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # Don't require a GUI to generate graphs
from matplotlib import pyplot as plt
import networkx as nx
import re
plt.ion()

class Results(object):
    def __init__(self, training_res, listen_params, num_agents, num_env, welfare, difference, cost):
        self.training_res = training_res
        self.listen_params = listen_params
        self.get_trimmed_listen_params()
        self.welfare = welfare
        self.welfareDifference = difference
        self.welfareCost = cost
        self.num_agents = num_agents
        self.num_env = num_env
        self.layers = len(listen_params) / num_agents
        self.G = None
        self.CG = None

    def reset(self):
        self.get_trimmed_listen_params()
        self.graph_org()
        self.graph_collapsed_org()

    def get_trimmed_listen_params(self, cutoff=.05):
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

    def graph_org(self, vspace=200, hspace=200, layout=True):
        numenv = len(self.trimmed[0].flatten())
        self.G = nx.DiGraph()
        for i in range(numenv):
            self.G.add_node(i, color="b", name="E" + str(i), category="environment")
            if( layout ):
                self.G[i]["graphics"] = {i: {'x':hspace*i, 'y':0}}
                #nx.set_node_attributes(self.G, "graphics", {i: {'x':hspace*i, 'y':0}})
        hspace = hspace * numenv / float(self.num_agents)
        hoffset = -1 * (hspace / numenv) # We want to center the nodes above the env
        for aix, agent in enumerate(self.trimmed):
            nodenum = int(numenv + aix)
            prefix = aix % self.num_agents
            layer = aix / self.num_agents
            n = "A%d_%d" % (prefix, layer)
            self.G.add_node(nodenum, color='r', name=n, category="agent", layer=layer)
            nodex = hoffset + hspace*prefix
            nodey = vspace * (layer + 1)
            if( layout ):
                self.G[nodenum]["graphics"] = {i: {'x':nodex, 'y':nodey}}
               	#nx.set_node_attributes(self.G, "graphics", {nodenum: {'x':nodex, 'y':nodey}})
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
                    self.G.add_edge(int(dest), nodenum, width=float(weight),
                        weight=float(abs(weight)))
            if( layer > 0 ):
                predecessor = int(numenv + aix - self.num_agents)
                self.G.add_edge(predecessor, nodenum, width=0, weight=0)

    def graph_collapsed_org(self):
        numenv = len(self.trimmed[0].flatten())
        self.CG = nx.DiGraph()
        for i in range(numenv):
            self.CG.add_node(i, color="b", name="E" + str(i), category="environment")
        for aix in range(self.num_agents):
            nodenum = int(numenv + aix)
            self.CG.add_node(nodenum, color='r', name="A"+str(aix), category="agent")
        for aix, agent in enumerate(self.trimmed):
            nodenum = int(numenv + aix)
            prefix = aix % self.num_agents
            layer = aix / self.num_agents
            src = numenv + prefix
            for dest, weight in enumerate(agent.flatten()):
                if( layer > 0 ):
                    dest += self.num_env
                if( abs(weight) > 0 and not self.CG.has_edge(int(dest), int(src)) ):
                    self.CG.add_edge(int(dest), int(src))

    # NetworkX always saves an int "label" in gml files
    # That's nice, but it prevents Cytoscape from reading our "name" attribute,
    # so we'll strip the labels out from the gml.
    def patch_gml(self, filename):
        with open(filename, "r+") as f:
            content = f.read()
            #newcontent = re.sub("label (\S+)", r'label "\1"', content)
            newcontent = re.sub("label (\S+)\s+", r'', content)
            f.seek(0, 0)
            f.truncate()
            f.write(newcontent)

    def graph_cytoscape(self, filename):
        if( self.G == None ):
            self.graph_org()
        #nx.write_graphml(self.G, filename)
        nx.write_gml(self.G, filename)
        self.patch_gml(filename)

    def graph_collapsed_cytoscape(self, filename):
        if( self.CG == None ):
            self.graph_collapsed_org()
        #nx.write_graphml(self.G, filename)
        nx.write_gml(self.CG, filename)
        self.patch_gml(filename)

    # Returns std-deviation of agent degree
    # Note: AGENT degree, not NODE degree
    def get_degree_distribution(self):
        if( self.G == None ):
            self.graph_org()
        numenv = len(self.trimmed[0].flatten())
        degrees = []
        for a in range(self.num_agents):
            degree = 0
            a += numenv
            while( a < len(self.G) ):
                degree += self.G.degree(a)
                a += self.num_agents
            degree /= float(self.layers)
            degrees.append(degree)
        return np.std(np.array(degrees))

    # Returns global reaching centrality for a weighted directed graph
    def global_reaching_centrality(self):
        centrality = []
        for i in range(self.num_agents):
            centrality.append(self.get_centrality_of_agent(i))
        max_centrality = max(centrality)
        total = 0.0
        for c in centrality:
            total += (max_centrality - c)
        return total / self.num_agents

    # Returns the sum of the centrality for each layer of the agent
    # NOTE: Mean is inappropriate here, because if a layer 2 is unused
    # then the (common) outlier will drop the centrality a ton
    def get_centrality_of_agent(self, agent):
        centrality = 0.0
        total_nodes = len(self.trimmed)
        while( agent < total_nodes ):
            centrality += self.get_centrality_of_node(agent)
            agent += self.num_agents
        return centrality

    # Returns the centrality of a node in a weighted graph
    def get_centrality_of_node(self, node):
        total_weight = 0.0
        total_nodes = len(self.trimmed)
        for i in range(total_nodes):
            path = self.get_listening_to_path(node, i)
            if( len(path) > 0 ):
                weight = self.get_weight_of_listening_to_path(node, i)
                total_weight += (weight / len(path))
        return ((1/float(total_nodes - 1)) * total_weight)

    # Returns a list of the agent IDs this agent listens to
    def get_agents_listened_to(self, agent):
        neighbors = []
        weights = self.trimmed[agent].flatten()
        layer = agent / self.num_agents
        agent_num = agent % self.num_agents
        if( layer == 0 ):
            return [] # Layer 0 only listens to env
        for i in range(len(weights)):
            # We (effectively) get a free path from our predecessor
            if( weights[i] != 0 or i == agent_num ):
                neighbors.append(i + (layer * self.num_agents))
        return neighbors

    # More expensive - get a list of everyone listening to this agent
    def get_agents_listening_to(self, agent):
        neighbors = []
        layer = agent / self.num_agents
        nextLayerStart = (layer+1)*self.num_agents
        agent_num = agent % self.num_agents
        # Outermost layer won't have anyone listening to it
        if( layer == self.layers-1 ):
            return []
        for a in range(nextLayerStart, nextLayerStart + self.num_agents):
            weights = self.trimmed[a].flatten()
            a_num = a % self.num_agents
            # We (effectively) get a free path to our predecessor
            if( weights[agent_num] != 0 or a_num == agent_num ):
                neighbors.append(a)
        return neighbors

    # Returns sum of all weights along the path, or zero if path impossible
    def get_weight_of_listened_to_path(self, src, dst):
        weight = 0.0
        path = self.get_listened_to_path(src, dst)
        #print "Received path: " + str(path)
        while( len(path) > 1 ):
            step = path.pop()
            nextAgent = path[-1] % self.num_agents
            #print "Looking at how much %d listens to %d" % (step, path[-1])
            weight += self.trimmed[step].flatten()[nextAgent]
        return weight

    # Returns sum of all weights along the path, or zero if path impossible
    def get_weight_of_listening_to_path(self, src, dst):
        weight = 0.0
        path = self.get_listening_to_path(src, dst)
        #print "Received path: " + str(path)
        while( len(path) > 1 ):
            step = path.pop()
            nextAgent = path[-1] % self.num_agents
            #print "Looking at how much %d listens to %d" % (step, path[-1])
            weight += self.trimmed[step].flatten()[nextAgent]
        return weight

    # Performs a simple breadth-first search to return path between agents
    # Iterative, so we won't hit max stack depth on a big graph
    def get_listened_to_path(self, src, dst):
        seen = set([src])
        parents = dict()
        q = self.get_agents_listened_to(src)
        while( len(q) > 0 ):
            c = q.pop(0)
            seen.add(c)
            if( c == dst ):
                path = [c]
                while( c in parents.keys() ):
                    c = parents[c]
                    path.append(c)
                path.append(src)
                path.reverse()
                return path
            for n in self.get_agents_listened_to(c):
                if( n not in seen ):
                    seen.add(n)
                    parents[n] = c
                    q.append(n)
        return []
            
    # Performs a simple breadth-first search to return path between agents
    # Iterative, so we won't hit max stack depth on a big graph
    def get_listening_to_path(self, src, dst):
        seen = set([src])
        parents = dict()
        q = self.get_agents_listening_to(src)
        while( len(q) > 0 ):
            c = q.pop(0)
            seen.add(c)
            if( c == dst ):
                path = [c]
                while( c in parents.keys() ):
                    c = parents[c]
                    path.append(c)
                path.append(src)
                path.reverse()
                return path
            for n in self.get_agents_listening_to(c):
                if( n not in seen ):
                    seen.add(n)
                    parents[n] = c
                    q.append(n)
        return []

    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
