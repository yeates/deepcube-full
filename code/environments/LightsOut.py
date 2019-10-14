import numpy as np
from random import choice
import argparse
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import time

import sys
sys.path.append('./')
from ml_utils import search_utils
from ml_utils import nnet_utils
import re

class LightsOut:
    def tf_dtype(self):
        import tensorflow as tf
        return(tf.uint8)

    def __init__(self, N):
        self.N = N

        self.legalPlays = [str(i) for i in range(N*N)]
        self.legalPlays_rev = [str(i) for i in range(N*N)]

        self.dtype = np.uint8
        self.solvedState = np.zeros(N*N, dtype=self.dtype)

        self.move_matrix = np.zeros((N*N, 5), dtype=np.int64)
        for move in range(N*N):
            xPos = int(np.floor(move / N))
            yPos = move % N

            right = move + N if xPos < (N-1) else move
            left = move - N if xPos > 0 else move
            up = move + 1 if yPos < (N-1) else move
            down = move - 1 if yPos > 0 else move

            self.move_matrix[move] = [move, right, left, up, down]
        
    def next_state(self, states_input, move):
        outputs = np.atleast_2d(states_input.copy())
        move = int(move)
        outputs[:, self.move_matrix[move]] = (outputs[:, self.move_matrix[move]] + 1) % 2

        return outputs


    def checkSolved(self, states):
        states = np.atleast_2d(states)
        return np.all(states == 0, axis=1)

    def getReward(self, states, isSolved = None):
        states = np.atleast_2d(states)
        #if type(isSolved) == type(None):
        #    isSolved = self.checkSolved(states)
        #    reward = 1.0*isSolved + (-1.0)*(np.logical_not(isSolved))

        reward = np.ones(shape=(states.shape[0]))
        return reward

    def state_to_nnet_input(self, states, randTransp=False):
        states_nnet = np.atleast_2d(states.copy())
        #states_nnet = states_nnet.reshape([states_nnet.shape[0],self.N,self.N,1])
        return(states_nnet)

    def generate_envs(self,numCubes,scrambleRange,probs=None,returnMoves=False):
        assert(scrambleRange[0] >= 0)
        scrambs = range(scrambleRange[0],scrambleRange[1]+1)
        legal = self.legalPlays
        states = []

        scrambleNums = np.zeros([numCubes],dtype=int)
        moves_all = []
        for cubeNum in range(numCubes):
            scrambled = np.array([self.solvedState])

            # Get scramble Num
            scrambleNum = np.random.choice(scrambs,p=probs)
            scrambleNums[cubeNum] = scrambleNum

            # Scramble cube
            moves = []
            for i in range(scrambleNum):
                move = choice(legal)
                scrambled = self.next_state(scrambled, move)
                moves.append(move)

            states.append(scrambled[0])
            moves_all.append(moves)

        if returnMoves:
            return(states,scrambleNums,moves_all)
        else:
            return(states,scrambleNums)

    def print_state(self, state):
        out = str(np.reshape(state, (self.n, self.n)))
        out = out.replace("1", "X")
        out = out.replace("0", "O")
        out = out.replace("[", " ")
        out = out.replace("]", " ")

        print(out)

class InteractiveEnv(plt.Axes):
    def __init__(self,state,env,heuristicFn=None):
        self.state = state
        self.env = env
        self.heuristicFn = heuristicFn
        self.N = np.sqrt(state.shape[1])

        super(InteractiveEnv, self).__init__(plt.gcf(),[0,0,1,1])

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        self.figure.canvas.mpl_connect('key_press_event',self._keyPress)

        self._updatePlot()

        self.move = []

    
    def _keyPress(self, event):
        if event.key.isdigit():
            self.move.append(event.key)
        elif event.key == 'enter':
            move = "".join(self.move)
            if move in self.env.legalPlays:
                self.state = self.env.next_state(self.state,move)
                self._updatePlot()
                if self.env.checkSolved(self.state)[0]:
                    print("SOLVED!")
            else:
                print("ERROR: %s is not a valid move" % (move))
            
            self.move = []
        elif event.key.upper() == 'N':
            self.stepNnet()
        elif event.key.upper() == 'S':
            self.solveNnet()


    def _updatePlot(self):
        self.clear()
        for squareIdx,square in enumerate(self.state[0]):
            color = 'darkgreen' if square == 0 else 'lightgreen'
            xPos = int(np.floor(squareIdx / self.N))
            yPos = squareIdx % self.N

            self.add_patch(patches.Rectangle((xPos/float(self.N),yPos/float(self.N)),1.0/self.N,1.0/self.N,0,linewidth=1,edgecolor='k',facecolor=color))
        self.figure.canvas.draw()

    
    def stepNnet(self):
        search = search_utils.BFS(self.state, self.heuristicFn, self.env)
        values, nextStatesValueReward = search.run(1)

        nextMoves = np.argmin(nextStatesValueReward,axis=1)

        self.state = self.env.next_state(self.state,self.env.legalPlays[nextMoves[0]])
        self._updatePlot()

    def solveNnet(self):
        startTime = time.time()

        BestFS_solve = search_utils.BestFS_solve(self.state,self.heuristicFn,self.env,bfs=0)
        isSolved, solveSteps, nodesGenerated_num = BestFS_solve.run(numParallel=100,depthPenalty=0.1,verbose=True)

        ### Make move
        moves = solveSteps[0]
        print("Neural network found solution of length %i (%s)" % (len(moves),time.time()-startTime))
        for move in moves:
            self.state = self.env.next_state(self.state,move)
            self._updatePlot()
            time.sleep(0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=5, help="")
    parser.add_argument('--heur', type=str, default=None, help="")
    parser.add_argument('--init', type=str, default=None, help="")
    args = parser.parse_args()

    env = LightsOut(args.N)
    if args.init is None:
        state = np.array([env.solvedState])
    else:
        state = np.array([[int(x) for x in list(re.sub("[^0-9]","",args.init))]])
    #state = np.array(env.generate_envs(1, [100, 100])[0])

    heuristicFn = None
    if args.heur is not None:
        heuristicFn = nnet_utils.loadNnet(args.heur,"",False,env)

    fig = plt.figure(figsize=(5, 5))
    interactiveEnv = InteractiveEnv(state,env,heuristicFn)
    fig.add_axes(interactiveEnv)

    plt.show()
