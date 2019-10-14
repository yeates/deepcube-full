import pdb
import argparse
import pickle
import numpy as np

opps = {'U':'D','L':'R','B':'F'}
for key in opps.keys():
    opps[opps[key]] = key

def moveToString(move):
    moveString = "".join([str(x) for x in move])
    return(moveString)

def simplifySoln(soln):
    newSoln = []
    idx = 0
    while idx < len(soln):
        move = soln[idx]
        if idx < (len(soln)-1):
            if move == soln[idx+1]:
                newSoln.append("%s%i" % (move[0],2))
                idx += 2
                continue
            elif opps[move[0]] == soln[idx+1][0]:
                if not (idx < (len(soln)-2)) or not (soln[idx+1] == soln[idx+2]):
                    if move[0] < soln[idx+1][0]:
                        newSoln.append("%s%s%s%s" % (move[0],move[1:],soln[idx+1][0],soln[idx+1][1:]))
                    else:
                        newSoln.append("%s%s%s%s" % (soln[idx+1][0],soln[idx+1][1:],move[0],move[1:]))
                    idx += 2
                    continue

        newSoln.append(move)
        idx += 1

    return(newSoln)

parser = argparse.ArgumentParser()
parser.add_argument('--soln1', type=str, required=True, help="")
parser.add_argument('--soln2', type=str, required=True, help="")

args = parser.parse_args()

data1 = pickle.load(open(args.soln1,"rb"))
data2 = pickle.load(open(args.soln2,"rb"))

solnsToSym = {
                'U1':'U-1','D1':'D-1','F1':'F-1','B1':'B-1','L1':'R-1','R1':'L-1',
                'U2':'U2','D2':'D2','F2':'F2','B2':'B2','L2':'R2','R2':'L2',
                'D1U1':'D-1U-1','B1F1':'B-1F-1','L1R1':'L-1R-1',
                'D1U-1':'D-1U1','B-1F1':'B1F-1','L-1R1':'L1R-1'
             }
for key in solnsToSym.keys():
    solnsToSym[solnsToSym[key]] = key

lim = len(data1["solutions"][data1["solutions"].keys()[0]])

solns1 = np.array(data1["solutions"][data1["solutions"].keys()[0]][:lim])
solns2 = np.array(data2["solutions"][data2["solutions"].keys()[0]][:lim])
times1 = np.array(data1["times"][data1["times"].keys()[0]][:lim])
times2 = np.array(data2["times"][data2["times"].keys()[0]][:lim])
nodesGenerated_num1 = np.array(data1["nodesGenerated_num"][data1["nodesGenerated_num"].keys()[0]][:lim])
nodesGenerated_num2 = np.array(data2["nodesGenerated_num"][data2["nodesGenerated_num"].keys()[0]][:lim])

lens1 = np.array([len(x) for x in solns1][:lim])
lens2 = np.array([len(x) for x in solns2][:lim])

solns1 = [[moveToString(move) for move in soln] for soln in solns1]
solns2 = [[moveToString(move) for move in soln] for soln in solns2]

solnsMod1 = [simplifySoln(soln) for soln in solns1]
solnsMod2 = [simplifySoln(soln) for soln in solns2]
#solnsMod1 = solns1
#solnsMod2 = solns2

isSymm = np.zeros(len(solnsMod1),dtype=np.bool)
idx = -1
for soln1,soln2 in zip(solnsMod1,solnsMod2):
    idx += 1
    
    if len(soln1) != len(soln2):
        continue
    misMatch = False
    for move1,move2 in zip(soln1,soln2):
        if move1 != move2 and solnsToSym[move1] != move2:
            misMatch = True
            break

    if not misMatch:
        isSymm[idx] = True

percentSymm = 100*float(sum(isSymm))/len(isSymm)
percentSameLen = 100*float(sum(lens1==lens2))/len(lens1)
percentSameLenNotSymm = 100*float(sum(lens1[np.invert(isSymm)]==lens2[np.invert(isSymm)]))/len(lens1[np.invert(isSymm)])
print("%.2f%% symmetric\n%.2f%% same solution length\n%.2f%% of nonsymmetric same solution length" % (percentSymm,percentSameLen,percentSameLenNotSymm))

