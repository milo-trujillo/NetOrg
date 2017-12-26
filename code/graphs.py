#!/usr/bin/env python
from results import Results
import pickle, os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

'''
    This is a standalone script, used to produce graphs of
    (parameter-we're-adjusting) vs (welfare)
    This is particularly useful for graphing the results of a parameter sweep.

    To use, define the folder where the data is, the name for the graph axis,
    and run.
'''

folder = "sc"
axis = "Speaking Cost"

plt.ion()
welfarefig = plt.figure()
welfareax = welfarefig.add_subplot(1,1,1)
grcfig = plt.figure()
grcax = grcfig.add_subplot(1,1,1)
degreefig = plt.figure()
degreeax = degreefig.add_subplot(1,1,1)
i = 0
startVal = 2
for fname in os.listdir(folder):
    print "Looking at " + fname
    if( os.path.isfile(folder + "/" + fname) and fname.endswith("pickle") ):
        print "Loading " + fname
        res = pickle.load(open(folder + "/" + fname))
        welfareax.plot(np.log(res.training_res), label="Iteration " + str(i+1))
        grcax.plot([startVal + i], [res.global_reaching_centrality()], "ro")
        degreeax.plot([startVal + i], [res.get_degree_distribution()], "ro")
        i += 1
welfareax.set_title("Trials")
welfareax.set_xlabel("Training Epoch")
welfareax.set_ylabel("Log(Welfare)")
welfareax.legend()
welfarefig.savefig("trials.png")
grcax.set_title("Centrality Parameter Sweep")
grcax.set_xlabel(axis)
grcax.set_ylabel("Global Reaching Centrality")
grcax.legend()
grcfig.savefig("centrality_sweep.png")
degreeax.set_title("Centrality Parameter Sweep")
degreeax.set_xlabel(axis)
degreeax.set_ylabel("Degree Std-Deviation (per agent)")
degreeax.legend()
degreefig.savefig("degree_sweep.png")
