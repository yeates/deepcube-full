import argparse
import sys
import time
import numpy as np

sys.path.append('./')
from ml_utils import nnet_utils
from ml_utils import search_utils
from environments import env_utils

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3, cube4")

parser.add_argument('--num_states', type=int, default=100, help="Number of cubes to solve per scramble depth")
parser.add_argument('--min_s', type=int, default=1, help="")
parser.add_argument('--max_s', type=int, default=30, help="")

parser.add_argument('--method', type=str, default="BFS", help="Method to use: bfs, bestfs, mcts, or kociemba")
parser.add_argument('--max_turns', type=int, default=None, help="Max number of turns before an episode ends")
parser.add_argument('--search_depth', type=int, default=1, help="Depth of nnet search")

parser.add_argument('--model_loc', type=str, default="", help="Location of model")
parser.add_argument('--model_name', type=str, default="model.meta", help="Which model to load")
parser.add_argument('--num_rollouts', type=int, default=10, help="Number of rollouts to do in MCTS")
parser.add_argument('--depth_penalty', type=float, default=0.1, help="")

parser.add_argument('--verbose', action='store_true', default=False, help="Print status to screen if true")
parser.add_argument('--noGPU', action='store_true', default=False, help="")
args = parser.parse_args()

Environment = env_utils.getEnvironment(args.env)

modelLoc = args.model_loc
numStates = args.num_states
maxTurns = args.max_turns
searchDepth = args.search_depth
numRollouts = args.num_rollouts
solveMethod = args.method.upper()
verbose = args.verbose

if maxTurns is None:
    maxTurns = args.max_s

load_start_time = time.time()
if solveMethod == "BFS" or solveMethod == "MCTS" or solveMethod == "MCTS_SOLVE" or solveMethod == "BESTFS":
    assert(modelLoc != "")
    ### Restore session
    heuristicFn = nnet_utils.loadNnet(args.model_loc,args.model_name,not args.noGPU,Environment)

print("Loaded: %s" % (time.time() - load_start_time))

### Run network on different scrambles
scrambleTests = range(args.min_s,args.max_s,1)
if args.max_s - args.min_s > 30:
    scrambleTests = np.linspace(args.min_s,args.max_s,30,dtype=np.int)

for scrambleNum in scrambleTests:
    solve_start_time = time.time()
    # Solve cubes
    testStates_cube, _ = Environment.generate_envs(numStates,[scrambleNum,scrambleNum])
    testStates = Environment.state_to_nnet_input(np.stack(testStates_cube))
    if solveMethod == "BFS" or solveMethod == "MCTS":
        stateVals = heuristicFn(np.stack(testStates_cube,axis=0))

        isSolved, solveSteps = search_utils.solve(testStates_cube,heuristicFn,Environment,maxTurns=maxTurns,searchDepth=searchDepth,numRollouts=numRollouts,searchMethod=solveMethod,verbose=verbose)
        
        percentSolved = 100*float(sum(isSolved))/float(len(isSolved))
        avgSolveSteps = 0.0
        if percentSolved > 0.0:
            avgSolveSteps = np.mean(solveSteps[isSolved])

    elif solveMethod == "BESTFS":
        stateVals = heuristicFn(np.stack(testStates_cube,axis=0))

        BestFS_solve = search_utils.BestFS_solve(testStates_cube,heuristicFn,Environment)
        isSolved, solveSteps = BestFS_solve.run(numParallel=numRollouts,depthPenalty=args.depth_penalty,verbose=verbose)

        percentSolved = 100*float(sum(isSolved))/float(len(isSolved))
        avgSolveSteps = 0.0
        if percentSolved > 0.0:
            solveSteps_isSolved = [x for x,y in zip(solveSteps,isSolved) if y == True]
            avgSolveSteps = np.mean([len(x) for x in solveSteps_isSolved])

    solve_elapsed_time = time.time() - solve_start_time
    print("Scramble Num: %i, perSolved: %.2f, avgSolveSteps: %.2f, StateVals: %.2f(%.2f), time: %.2f" % (scrambleNum, percentSolved, avgSolveSteps,np.mean(stateVals),np.std(stateVals),solve_elapsed_time))

