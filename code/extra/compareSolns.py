import argparse
import pickle
import numpy as np

def printStats(data,hist=False):
    print("Min/Max/Median/Mean(Std) %f/%f/%f/%f(%f)" % (min(data),max(data),np.median(data),np.mean(data),np.std(data)))
    if hist:
        hist1 = np.histogram(data)
        for x,y in zip(hist1[0],hist1[1]):
            print("%s %s" % (x,y))

parser = argparse.ArgumentParser()
parser.add_argument('--soln1', type=str, required=True, help="")
parser.add_argument('--soln2', type=str, required=True, help="")

args = parser.parse_args()

data1 = pickle.load(open(args.soln1,"rb"))
data2 = pickle.load(open(args.soln2,"rb"))

solns1 = np.array(data1["solutions"][data1["solutions"].keys()[0]])
solns2 = np.array(data2["solutions"][data2["solutions"].keys()[0]])

notNoneIdxs = [x != None for x in solns1]

solns1 = solns1[notNoneIdxs]
solns2 = solns2[notNoneIdxs]

lens1 = np.array([len(x) for x in solns1])
lens2 = np.array([len(x) for x in solns2])

print("\n--Optimal Solutions---")
print("-Lengths-")
printStats(lens1)

print("\n--DeepCubeA---")
print("-Lengths-")
printStats(lens2)

print("\n\n------DeepCubeA % Optimal-----")
perOptimal = 100*sum(lens1 == lens2)/float(len(lens1))
print("%.2f%%" % (perOptimal))
