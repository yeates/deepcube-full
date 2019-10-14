import numpy as np
import pdb
import os
import sys
import subprocess
import re

class Optimal:
    """
        colors: Cube representation used for simulator
        Returns: Moves in representation used for simulator
    """
    @staticmethod
    def solve(states,startIdx=0,endIdx=-1):

        ### Write input file
        inputFileName = "./solvers/puzzle15/ida/input.txt"
        inputFile = open(inputFileName,'w')
        for state in states:
            state_str = " ".join([str(x) for x in state])
            inputFile.write("%s\n" % (state_str))
        inputFile.close()

        ### Run method
        os.system('cd ./solvers/puzzle15/ida/ && stdbuf -o0 ./main input.txt > output.txt 2>&1')

        ### Parse output
        with open("./solvers/puzzle15/ida/output.txt") as outputFile:
            output = outputFile.readlines()

        elapsedTimes = [x.strip() for x in output if 'Total time:' in x]
        nodesGenerated_nums = [x.strip() for x in output if 'Total nodes generated:' in x]
        solns = [x.strip() for x in output if 'Moves are:' in x]

        elapsedTimes = [float(re.search('Total time:\s+(\S+)',x).group(1)) for x in elapsedTimes]
        nodesGenerated_nums = [int(re.search('Total nodes generated:\s+(\S+)',x).group(1)) for x in nodesGenerated_nums]
        solns = [(re.search('Moves are:\s+(.+)',x).group(1)).split(" ") for x in solns]

        return(solns,nodesGenerated_nums,elapsedTimes)

