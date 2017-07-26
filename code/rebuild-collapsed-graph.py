#!/usr/bin/env python
import pickle, sys
from results import Results

if __name__ == "__main__":
    if( len(sys.argv) != 3 ):
        print "USAGE: %s <results.pickle> <output.graphml>" % sys.argv[0]
        sys.exit(1)
    res = pickle.load(open(sys.argv[1], "rb"))
    res.reset()
    res.graph_collapsed_cytoscape(sys.argv[2])
