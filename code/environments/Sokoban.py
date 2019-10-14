import numpy as np
from random import choice
import argparse
import matplotlib.pyplot as plt

import os
import time
import cPickle as pickle

import sys
sys.path.append('./')
from ml_utils import search_utils
from ml_utils import nnet_utils

#import gym
#import gym_sokoban

#dir(gym_sokoban)

class Sokoban:
    def tf_dtype(self):
        import tensorflow as tf
        return(tf.uint8)

    def __init__(self, dim, numBoxes):
        self.dim = dim
        self.numBoxes = numBoxes

        if self.dim == 10 and self.numBoxes == 4:
            self.env_name = 'Sokoban-v1'

        self.legalPlays = np.array(['a','s','d','w'])
        self.legalPlays_rev = np.array(['d','w','a','s'])

        self.dtype = np.uint8

        #self.env = gym.make(self.env_name)

        numPos = self.dim ** 2
        self.wallBegin = 0; self.wallEnd = numPos
        self.goalBegin = numPos; self.goalEnd = 2*numPos
        self.boxBegin = 2*numPos; self.boxEnd = 3*numPos
        self.sokobanBegin = 3*numPos; self.sokobanEnd = 4*numPos

        self.numPos = numPos

        self.nextStateIdxs = np.zeros((self.numPos,len(self.legalPlays)),dtype=np.int)
        for d1 in range(self.dim):
            for d2 in range(self.dim):
                for moveIdx,move in enumerate(self.legalPlays):
                    coord = np.array([d1,d2])
                    coord_next = coord.copy()
                    if move.lower() == 'a':
                        coord_next[1] = (coord[1] - 1)*(coord[1] != 0)
                    elif move.lower() == 'd':
                        coord_next[1] = np.minimum(coord[1] + 1,self.dim-1)
                    elif move.lower() == 's':
                        coord_next[0] = np.minimum(coord[0] + 1,self.dim-1)
                    elif move.lower() == 'w':
                        coord_next[0] = (coord[0] - 1)*(coord[0] != 0)

                    coord_flat = coord[0]*self.dim + coord[1]
                    coord_flat_next = coord_next[0]*self.dim + coord_next[1]
                    self.nextStateIdxs[coord_flat,moveIdx] = coord_flat_next

    def get_new_game_state(self):
        """
        self.env.reset()
        room_fixed_flat = self.env.room_fixed.flatten()
        room_state_flat = self.env.room_state.flatten()
        
        wall_idxs = 1*(room_fixed_flat == 0)
        goal_idxs = 1*(room_fixed_flat == 2)
        box_idxs = 1*(room_state_flat == 4)
        sokoban_idxs = 1*(room_state_flat == 5)

        state = np.concatenate((wall_idxs,goal_idxs,box_idxs,sokoban_idxs),axis=0)
        
        state = np.array([state])

        state = state.astype(self.dtype)
        """
        statesDir = "environments/sokoban_utils/sokoban_10_10_4/"
        stateFiles = [f for f in os.listdir(statesDir) if os.path.isfile(os.path.join(statesDir, f)) and ('.pkl' in f)]
        states = pickle.load(open("%s/%s" % (statesDir,choice(stateFiles)),"rb"))
        idx = choice(range(states.shape[0]))
        state = states[idx:(idx+1),:]
        return(state)

    def parse_states(self,fileName):
        states = []
        lines = [line.rstrip('\n') for line in open(fileName)]
        rowIdx = -1
        state = -np.ones(self.numPos*4)
        for line in lines:
            if rowIdx >= 0:
                startIdx = rowIdx*self.dim
                endIdx = (rowIdx+1)*self.dim
                state[(self.wallBegin+startIdx):(self.wallBegin+endIdx)] = np.array([x == "#" for x in line])*1
                state[(self.goalBegin+startIdx):(self.goalBegin+endIdx)] = np.array([x == "." for x in line])*1
                state[(self.boxBegin+startIdx):(self.boxBegin+endIdx)] = np.array([x == "$" for x in line])*1
                state[(self.sokobanBegin+startIdx):(self.sokobanBegin+endIdx)] = np.array([x == "@" for x in line])*1
                rowIdx = rowIdx + 1
            if ";" in line:
                rowIdx = 0
                state = -np.ones(self.numPos*4)
                continue


            if rowIdx == self.dim:
                assert(sum(state[self.goalBegin:self.goalEnd]) == self.numBoxes)
                assert(sum(state[self.boxBegin:self.boxEnd]) == self.numBoxes)
                assert(sum(state[self.sokobanBegin:self.sokobanEnd]) == 1)
                states.append(state.copy())
                rowIdx = -1


        states = np.stack(states,axis=0)
        states = states.astype(self.dtype)
        return(states)

    def render(self,states):
        wall_idxs = states[:,self.wallBegin:self.wallEnd]
        goal_idxs = states[:,self.goalBegin:self.goalEnd]
        box_idxs = states[:,self.boxBegin:self.boxEnd]
        sokoban_idxs = states[:,self.sokobanBegin:self.sokobanEnd]

        states_rendered = np.ones((states.shape[0],self.numPos))

        states_rendered[wall_idxs == 1] = 0
        states_rendered[goal_idxs == 1] = 2
        states_rendered[box_idxs == 1] = 4
        states_rendered[sokoban_idxs == 1] = 5

        states_rendered[(goal_idxs == 1) & (sokoban_idxs == 1)] = 3
        states_rendered[(goal_idxs == 1) & (box_idxs == 1)] = 6

        return(states_rendered)

    def next_state(self, states_input, move, reverse=False):
        moveIdx = np.where(self.legalPlays == np.array(move))[0][0]

        outputs = np.atleast_2d(states_input.copy())
        numStates = outputs.shape[0]
        
        # Move sokoban
        _, sokobanIdxs = np.where(outputs[range(numStates),self.sokobanBegin:self.sokobanEnd] == 1)
        sokobanIdxs_next = self.nextStateIdxs[sokobanIdxs,moveIdx].copy()
        
        # Check if hitting a wall
        hitWall = outputs[range(numStates),self.wallBegin + sokobanIdxs_next] == 1
        sokobanIdxs_next[hitWall] = sokobanIdxs[hitWall]

        if reverse:
            moveIdx_rev = np.where(self.legalPlays == self.legalPlays_rev[moveIdx])[0][0]
            sokobanIdxs_next_rev = self.nextStateIdxs[sokobanIdxs,moveIdx_rev].copy()
            
            # Check if hitting a wall
            hitWallOrBox = (outputs[range(numStates),self.wallBegin + sokobanIdxs_next_rev] == 1) | (outputs[range(numStates),self.boxBegin + sokobanIdxs_next_rev] == 1)
            sokobanIdxs_next_rev[hitWallOrBox] = sokobanIdxs[hitWallOrBox]
            sokobanIdxs_next[hitWallOrBox] = sokobanIdxs[hitWallOrBox]


        # Check if box is pushed
        boxIdxs_ex, boxIdxs = np.where(outputs[range(numStates),self.boxBegin:self.boxEnd] == 1)
        boxIdxs_next = boxIdxs.copy()

        box_pushed_idxs = np.where(sokobanIdxs_next[boxIdxs_ex] == boxIdxs)[0]
        box_pushed_ex = boxIdxs_ex[box_pushed_idxs]
        if box_pushed_idxs.shape[0] > 0:
            boxIdxs_pushed = boxIdxs[box_pushed_idxs]

            # move any box that has been pushed
            if not reverse:
                boxIdxs_pushed_next = self.nextStateIdxs[boxIdxs_pushed,moveIdx].copy()

                # Check hitting a wall or another box
                hitWallOrBox = (outputs[box_pushed_ex,self.wallBegin + boxIdxs_pushed_next] == 1) | (outputs[box_pushed_ex,self.boxBegin + boxIdxs_pushed_next] == 1)

                boxIdxs_pushed_next[hitWallOrBox] = boxIdxs_pushed[hitWallOrBox]

                # if box has not been pushed then Sokoban does not move
                sokobanIdxs_next[box_pushed_ex] = sokobanIdxs_next[box_pushed_ex]*np.invert(hitWallOrBox) + sokobanIdxs[box_pushed_ex]*hitWallOrBox
            else:
                boxIdxs_pushed_next = self.nextStateIdxs[boxIdxs_pushed,moveIdx_rev].copy()



            boxIdxs_next[box_pushed_idxs] = boxIdxs_pushed_next

        outputs[range(numStates),self.sokobanBegin + sokobanIdxs] = 0
        if not reverse:
            outputs[range(numStates),self.sokobanBegin + sokobanIdxs_next] = 1
        else:
            outputs[range(numStates),self.sokobanBegin + sokobanIdxs_next_rev] = 1

        outputs[boxIdxs_ex,self.boxBegin + boxIdxs] = 0
        outputs[boxIdxs_ex,self.boxBegin + boxIdxs_next] = 1

        return outputs


    def checkSolved(self, states):
        states = np.atleast_2d(states)
        goal_idxs = states[:,self.goalBegin:self.goalEnd]
        box_idxs = states[:,self.boxBegin:self.boxEnd]

        return np.all(goal_idxs == box_idxs,axis=1)

    def getReward(self, states, isSolved = None):
        states = np.atleast_2d(states)
        reward = np.ones(shape=(states.shape[0]))
        return reward

    def state_to_nnet_input(self, states, randTransp=False):
        states_nnet = np.atleast_2d(states.copy())
        #states_nnet = states_nnet.reshape([states_nnet.shape[0],self.N,self.N,1])
        return(states_nnet)

    def get_pullable_idx(self,states):
        ### Get box idxs
        boxIdxs_ex, boxIdxs = np.where(states[:,self.boxBegin:self.boxEnd] == 1)
        boxIdxs = boxIdxs.reshape([states.shape[0],self.numBoxes])

        boxPulledIdxs = -np.ones([states.shape[0]],dtype=np.int)
        boxPulledIdxs_next = -np.ones([states.shape[0]],dtype=np.int)
        sokobanIdxs_next = -np.ones([states.shape[0]],dtype=np.int)
        ### Move sokoban to a place adjacent to box that it can pull
        for stateIdx in range(states.shape[0]):
            ### Get idxs adjacent to box
            for moveIdx in np.random.permutation(len(self.legalPlays)):
                boxAdjIdxs = self.nextStateIdxs[boxIdxs[stateIdx],moveIdx]
                boxAdjIdxs2 = self.nextStateIdxs[boxAdjIdxs,moveIdx]
                
                boxAdjIdxs12 = np.stack((boxAdjIdxs,boxAdjIdxs2))

                canPull = np.all((states[stateIdx,self.wallBegin + boxAdjIdxs12] != 1) & (states[stateIdx,self.boxBegin + boxAdjIdxs12] != 1),axis=0) & (boxAdjIdxs != boxAdjIdxs2)

                if max(canPull):
                    box_chose = np.random.choice(np.where(canPull)[0])

                    boxPulledIdxs[stateIdx] = boxIdxs[stateIdx][box_chose]
                    boxPulledIdxs_next[stateIdx] = boxAdjIdxs[box_chose]
                    sokobanIdxs_next[stateIdx] = boxAdjIdxs2[box_chose]
                    break

        return(boxPulledIdxs,boxPulledIdxs_next,sokobanIdxs_next)

    def get_reachable_boxes(self,states,sokobanIdxs):
        reachableBoxes = [None]*states.shape[0]
        

        return(reachableBoxes)
    
    def make_solved_state(self,states):
        states = np.atleast_2d(states)
        states_solved = states.copy()

        numStates = states_solved.shape[0]

        _, sokobanIdxs = np.where(states_solved[:,self.sokobanBegin:self.sokobanEnd] == 1)

        ### Set boxes to goal
        states_solved[:,self.boxBegin:self.boxEnd] = states_solved[:,self.goalBegin:self.goalEnd]
        
        ### Set sokoban idx to pullable idx
        _, sokobanIdxs_solved, _ = self.get_pullable_idx(states_solved)

        states_solved[range(numStates),self.sokobanBegin + sokobanIdxs] = 0
        states_solved[range(numStates),self.sokobanBegin + sokobanIdxs_solved] = 1

        return(states_solved)

    def pull_box(self,states):
        states = np.atleast_2d(states)
        states_pulled = states.copy()

        numStates = states_pulled.shape[0]
        _, sokobanIdxs = np.where(states_pulled[:,self.sokobanBegin:self.sokobanEnd] == 1)

        boxPulledIdxs, boxPulledIdxs_next, sokobanIdxs_next = self.get_pullable_idx(states_pulled)

        states_pulled[range(numStates),self.sokobanBegin + sokobanIdxs] = 0
        states_pulled[range(numStates),self.sokobanBegin + sokobanIdxs_next] = 1

        states_pulled[range(numStates),self.boxBegin + boxPulledIdxs] = 0
        states_pulled[range(numStates),self.boxBegin + boxPulledIdxs_next] = 1

        return(states_pulled)

    def generate_envs(self,numStates,scrambleRange,probs=None):
        assert(scrambleRange[0] >= 0)
        scrambs = range(scrambleRange[0],scrambleRange[1]+1)
        states = []

        ### Load states
        statesDir = "environments/sokoban_utils/sokoban_10_10_4/"
        stateFiles = [f for f in os.listdir(statesDir) if os.path.isfile(os.path.join(statesDir, f)) and ('.pkl' in f)]
        while len(states) != numStates:
            numToLoad = numStates-len(states)

            stateFile = choice(stateFiles)
            states_load = pickle.load(open("%s/%s" % (statesDir,stateFile),"rb"))

            load_idxs = np.random.permutation(states_load.shape[0])
            load_idxs = load_idxs[:min(numToLoad,len(load_idxs))]

            [states.append(states_load[i]) for i in load_idxs]
        
        states = np.stack(states,axis=0)

        ### Take reverse steps
        scrambleNums = np.random.choice(scrambs,size=numStates,replace=True,p=probs)

        states = self.make_solved_state(states)
        
        numMoves = np.zeros(numStates)
        while (np.max(numMoves < scrambleNums) == True):
            poses = np.where((numMoves < scrambleNums))[0]
            
            subsetSize = max(len(poses)/len(self.legalPlays),1)
            poses = np.random.choice(poses,subsetSize)
            
            move = choice(self.legalPlays)

            states[poses] = self.next_state(states[poses],move,reverse=True);
            
            numMoves[poses] = numMoves[poses] + 1

        states = list(states)
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

        if self.state is None:
            self.state = self.env.get_new_game_state()

        super(InteractiveEnv, self).__init__(plt.gcf(),[0,0,1,1])

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        self.figure.canvas.mpl_connect('key_press_event',self._keyPress)

        self._updatePlot()

        self.move = []

    
    def _keyPress(self, event):
        if event.key.upper() in 'ASDW':
            self.state = self.env.next_state(self.state,event.key.lower())
            self._updatePlot()
            if self.env.checkSolved(self.state)[0]:
                print("SOLVED!")
        elif event.key.upper() in 'R':
            self.state = self.env.get_new_game_state()
            self._updatePlot()
        elif event.key.upper() in 'O':
            self.state = self.env.make_solved_state(self.state)
            self._updatePlot()
        elif event.key.upper() in 'P':
            for i in range(1000):
                self.state = self.env.next_state(self.state,choice(self.env.legalPlays),reverse=True)
            self._updatePlot()
        elif event.key.upper() == 'N':
            self.stepNnet()
        elif event.key.upper() == 'M':
            self.solveNnet()


    def _updatePlot(self):
        self.clear()
        renderedIm = self.env.render(self.state)
        renderedIm = renderedIm.reshape((self.env.dim,self.env.dim))
        self.imshow(renderedIm)
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
    parser.add_argument('--heur', type=str, default=None, help="")
    args = parser.parse_args()

    state = None
    env = Sokoban(10,4)
    #state = np.expand_dims(env.generate_envs(100, [1000, 1000])[0][0],0)

    heuristicFn = None
    if args.heur is not None:
        heuristicFn = nnet_utils.loadNnet(args.heur,"",False,env)

    fig = plt.figure(figsize=(5, 5))
    interactiveEnv = InteractiveEnv(state,env,heuristicFn)
    fig.add_axes(interactiveEnv)

    plt.show()
