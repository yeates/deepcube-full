import numpy as np
from random import choice
import re

import sys
sys.path.append('./')

def getEnvironment(envName):
    envName = envName.lower()
    if envName == 'cube3':
        from environments.cube_interactive_simple import Cube
        Environment = Cube(N=3,moveType="qtm")
    elif envName == 'cube3htm':
        from environments.cube_interactive_simple import Cube
        Environment = Cube(N=3,moveType="htm")
    elif envName == 'cube3htmaba':
        from environments.cube_interactive_simple import Cube
        Environment = Cube(N=3,moveType="htmaba")
    elif envName == 'cube4':
        from environments.cube_interactive_simple_4 import Cube as Environment
    elif envName == 'cube4d2':
        from environments.cube4D import Cube
        Environment = Cube(2)
    elif envName == 'puzzle15':
        from environments.puzzleN import PuzzleN
        Environment = PuzzleN(4)
    elif envName == 'puzzle24':
        from environments.puzzleN import PuzzleN
        Environment = PuzzleN(5)
    elif envName == 'puzzle35':
        from environments.puzzleN import PuzzleN
        Environment = PuzzleN(6)
    elif envName == 'puzzle48':
        from environments.puzzleN import PuzzleN
        Environment = PuzzleN(7)
    elif 'lightsout' in envName:
        from environments.LightsOut import LightsOut
        m = re.search('lightsout([\d]+)',envName)
        Environment = LightsOut(int(m.group(1)))
    elif 'hanoi' in envName:
        from environments.Hanoi import Hanoi
        m = re.search('hanoi([\d]+)d([\d]+)p',envName)
        numDisks = int(m.group(1))
        numPegs = int(m.group(2))

        Environment = Hanoi(numDisks,numPegs)
    elif envName == 'sokoban':
        from environments.Sokoban import Sokoban
        Environment = Sokoban(10,4)
        

    return(Environment)


def generate_envs(Environment,numPuzzles,scrambleRange,probs=None):
    assert(scrambleRange[0] > 0)
    scrambs = range(scrambleRange[0],scrambleRange[1]+1)
    legal = Environment.legalPlays
    puzzles = []
    puzzles_symm = []

    scrambleNums = np.zeros([numPuzzles],dtype=int)
    moves = []
    for puzzleNum in range(numPuzzles):
        startConfig_idx = np.random.randint(0,len(Environment.solvedState_all))

        scrambled = Environment.solvedState_all[startConfig_idx]
        scrambled_symm = np.stack(Environment.solvedState_all,axis=0)
        assert(Environment.checkSolved(scrambled))

        # Get scramble Num
        scrambleNum = np.random.choice(scrambs,p=probs)
        scrambleNums[puzzleNum] = scrambleNum
        # Scramble puzzle
        while Environment.checkSolved(scrambled): # don't return any solved puzzles
            moves_puzzle = []
            for i in range(scrambleNum):
                move = choice(legal)
                scrambled = Environment.next_state(scrambled, move)
                scrambled_symm = Environment.next_state(scrambled_symm, move)
                moves_puzzle.append(move)

        moves_puzzle.append(move)

        puzzles.append(scrambled)
        puzzles_symm.append(scrambled_symm)
    return(puzzles,scrambleNums,moves,puzzles_symm)


