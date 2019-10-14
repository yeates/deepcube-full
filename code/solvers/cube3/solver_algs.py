import numpy as np
import os
import sys
sys.path.append('./solvers/cube3/korf/')
import subprocess
import re

class Kociemba:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(colors,startIdx=0,endIdx=-1):
        ### Preprocess input
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        N = 3
        colors = colors/(N**2)
        kociembaOrder = np.array([],dtype=int)
        for face in [0,3,5,1,2,4]:
            for j in range(2,-1,-1):
                for i in range(0,3):
                    kociembaOrder = np.append(kociembaOrder,np.ravel_multi_index((face,i,j),(6,N,N)))

        colors_kociemba_ordered = colors[kociembaOrder]
        colors_to_kociemba_input = {0:'U', 1:'D', 2:'L', 3:'R', 4: 'B', 5: 'F'}
        kociemba_input = "".join([colors_to_kociemba_input[x] for x in colors_kociemba_ordered])

        ### Solve
        kociembaSoln = kociemba.solve(kociemba_input).split(" ")

        ### Convert solution
        moves = []
        for soln in kociembaSoln:
            direction = 1
            if '\'' in soln:
                direction = -1

            move = [str(soln[0]),direction]
            moves.append(move)
            if '2' in soln:
                moves.append(move)

        return(moves)

class Optimal:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(states,startIdx=0,endIdx=-1,combineOutputs=False,solveInputs=None):
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        ### Preprocess input
        if solveInputs is None:
            solveInputs = []
            for state in states:
                kocSoln = Kociemba.solve(state)
                kocSoln_backwards = kocSoln[::-1]
                solveInput = "".join([x[0] + "1" if x[1] == -1 else x[0] + "3" for x in kocSoln_backwards])
                solveInputs.append(solveInput)
        
        ### Solve cube
        inputFileName = "input1.txt"
        outputFileName = "output1.txt"

        inputFileLoc = "./solvers/cube3/rokicki/"
        inputFile = open("%s/%s" % (inputFileLoc,inputFileName),'w')
        for solveInput in solveInputs:
            inputFile.write("%s\n" % (solveInput))

        inputFile.close()

        os.system('cd ./solvers/cube3/rokicki/ && ./nxopt33 -t 40 - < %s > %s' % (inputFileName,outputFileName))

        with open("./solvers/cube3/rokicki/%s" % (outputFileName)) as outputFile:
            output = outputFile.readlines()
        solns_korf = [x.strip() for x in output if x[0] == ' '][1:]
        soln_stats = [x.strip() for x in output if 'Solved' in x]

        ### Add moves
        soln_to_move = {'U':'U', 'L':'L', 'R':'R', 'D':'D', 'B':'B', 'F':'F'}
        solns = []
        times = np.zeros(len(states))
        nodesGenerated_num = np.zeros(len(states))
        for soln_korf in solns_korf:
            moves = []
            soln_korf = soln_korf.split(" ")
            for move_korf in soln_korf:
                if len(move_korf) == 0:
                    continue

                direction = 1 # direction
                if '3' in move_korf:
                    direction = -1

                twice = False # check for 180 degree turn
                if '2' in move_korf:
                    twice = True
                    move_korf = move_korf[0]

                move = [soln_to_move[move_korf[0]],direction] # add move

                moves.append(move)
                if twice:
                    moves.append(move)
            solns.append(moves)

    
        for soln_stat in soln_stats:
            idx = int(re.search('Solved\s+([0-9]+)',soln_stat).group(1)) - 1
            nodesGenerated_num_idx = int(re.search('probes\s+([0-9]+)',soln_stat).group(1))
            time = float(re.search('time\s+(\S+)',soln_stat).group(1))

            nodesGenerated_num[idx] = nodesGenerated_num_idx
            times[idx] = time


        return(solns,nodesGenerated_num,times)

class Optimal_2:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(colors):
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        ### Preprocess input
        kocSoln = Kociemba.solve(colors)
        kocSoln_backwards = kocSoln[::-1]
        solveInput = " ".join([x[0] if x[1] == -1 else x[0] + "'" for x in kocSoln_backwards])
        
        ### Solve cube
        #print("Optimal solver input: %s" % (solveInput))
        result = subprocess.check_output(['./optiqtm',solveInput],cwd='./solvers/cube3/optimal/',stderr=subprocess.STDOUT)
        moves_korf = [x for x in result.split("\n")[-3].split(" ") if '(' not in x and ')' not in x]
        #print("Korf Moves: %s" % (moves_korf))

        ### Add moves
        soln_to_move = {'U':'U', 'L':'L', 'R':'R', 'D':'D', 'B':'B', 'F':'F'}
        moves = []
        for move_korf in moves_korf:
            if len(move_korf) == 0:
                continue

            direction = 1 # direction
            if '\'' in move_korf:
                direction = -1

            twice = False # check for 180 degree turn
            if '2' in move_korf:
                twice = True
                move_korf = move_korf[0]

            move = [soln_to_move[move_korf[0]],direction] # add move

            moves.append(move)
            if twice:
                moves.append(move)

 
        return(moves)

class Korf:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(colors):
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        ### Preprocess input
        kocSoln = Kociemba.solve(colors)
        kocSoln_backwards = kocSoln[::-1]
        twistInput = " ".join([x[0] if x[1] == -1 else x[0] + "'" for x in kocSoln_backwards])
        
        result = subprocess.check_output(['./twist.out',twistInput],cwd='./solvers/cube3/korf_2/',stderr=subprocess.STDOUT)
        korf_rep = result.split('\n')[-3]

        ### Solve cube
        #print("Korf Rep: %s" % (korf_rep))
        result = subprocess.check_output(['./a.out',korf_rep],cwd='./solvers/cube3/korf_2/',stderr=subprocess.STDOUT)
        moves_korf = [x for x in result.split("\n")[-3].split(" ") if '(' not in x and ')' not in x]
        #print("Korf Moves: %s" % (moves_korf))

        ### Add moves
        soln_to_move = {'U':'U', 'L':'L', 'R':'R', 'D':'D', 'B':'B', 'F':'F'}
        moves = []
        for move_korf in moves_korf:
            if len(move_korf) == 0:
                continue

            direction = 1 # direction
            if '\'' in move_korf:
                direction = -1

            twice = False # check for 180 degree turn
            if '2' in move_korf:
                twice = True
                move_korf = move_korf[0]

            move = [soln_to_move[move_korf[0]],direction] # add move

            moves.append(move)
            if twice:
                moves.append(move)

 
        return(moves)

class Korf_2:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(colors):
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        ### Preprocess input
        N = 3
        colors = colors/(N**2)

        korfOrder = np.array([],dtype=int)
        face = 0
        for j in range(2,-1,-1):
            for i in range(0,3):
                if i == 1 and j == 1:
                    continue
                korfOrder = np.append(korfOrder,np.ravel_multi_index((face,i,j),(6,N,N)))

        for j in range(2,-1,-1):
            for face in [2,5,3,4]:
                for i in range(0,3):
                    if i == 1 and j == 1:
                        continue
                    korfOrder = np.append(korfOrder,np.ravel_multi_index((face,i,j),(6,N,N)))

        face = 1
        for j in range(2,-1,-1):
            for i in range(0,3):
                if i == 1 and j == 1:
                    continue
                korfOrder = np.append(korfOrder,np.ravel_multi_index((face,i,j),(6,N,N)))

        colors_ordered = colors[korfOrder]
        colors_to_input = {0:'r', 1:'o', 2:'b', 3:'g', 4: 'y', 5: 'w'}
        cubeRep = "".join([colors_to_input[x] for x in colors_ordered])

        korfRep = cube_convert.convertRepresentation(cubeRep)
        #print("Korf Rep: %s" % (korfRep))

        ### Get solution
        result = subprocess.check_output(['./solver',korfRep],cwd='./solvers/cube3/korf/',stderr=subprocess.STDOUT)
        moves_korf = result.split('\n')[-2].split(" ")

        ### Add moves
        soln_to_move = {'T':'U', 'L':'L', 'R':'R', 'D':'D', 'B':'B', 'F':'F'}
        moves = []
        for move_korf in moves_korf:
            if len(move_korf) == 0:
                continue

            direction = 1 # direction
            if '\'' in move_korf:
                direction = -1

            twice = False # check for 180 degree turn
            if '2' in move_korf:
                twice = True
                move_korf = move_korf[1:]

            move = [soln_to_move[move_korf[0]],direction] # add move

            moves.append(move)
            if twice:
                moves.append(move)

 
        return(moves)


