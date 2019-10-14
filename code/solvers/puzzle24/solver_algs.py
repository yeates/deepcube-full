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
    def solve(states,startIdx=0,endIdx=-1,combineOutputs=False):
        if not combineOutputs:
            ### Write input file
            inputFileName_base = "input%i_%i.txt" % (startIdx,endIdx)
            outputFileName_base = "output%i_%i.txt" % (startIdx,endIdx)
            inputFileName = "./solvers/puzzle24/ida/%s" % (inputFileName_base)
            inputFile = open(inputFileName,'w')
            for state in states:
                state_str = " ".join([str(x) for x in state])
                inputFile.write("%s\n" % (state_str))
            inputFile.close()

            ### Run method
            os.system('cd ./solvers/puzzle24/ida/ && ./main %s > %s 2>&1' % (inputFileName_base,outputFileName_base))

            ### Parse output
            with open("./solvers/puzzle24/ida/%s" % (outputFileName_base)) as outputFile:
                output = outputFile.readlines()

            solns = [x.strip() for x in output if 'Moves are:' in x]
            nodesGenerated_nums = [x.strip() for x in output if 'Total nodes generated:' in x]
            elapsedTimes = [x.strip() for x in output if 'Total time:' in x]

            solns = [(re.search('Moves are:\s+(.+)',x).group(1)).split(" ") for x in solns]
            nodesGenerated_nums = [int(re.search('Total nodes generated:\s+(\S+)',x).group(1)) for x in nodesGenerated_nums]
            elapsedTimes = [float(re.search('Total time:\s+(\S+)',x).group(1)) for x in elapsedTimes]
        else:
            solns = []
            nodesGenerated_nums = []
            elapsedTimes = []
            noSolnNum = 0
            for i in range(startIdx,endIdx):
                fileName = "./solvers/puzzle24/ida/output%i_%i.txt" % (i,i+1)
                with open(fileName) as outputFile:
                    output = outputFile.readlines()

                soln = [x.strip() for x in output if 'Moves are:' in x]
                nodesGenerated_num = [x.strip() for x in output if 'Total nodes generated:' in x]
                elapsedTime = [x.strip() for x in output if 'Total time:' in x]

                if len(soln) == 1:
                    soln = (re.search('Moves are:\s+(.+)',soln[0]).group(1)).split(" ")
                    nodesGenerated_num = int(re.search('Total nodes generated:\s+(\S+)',nodesGenerated_num[0]).group(1))
                    elapsedTime = float(re.search('Total time:\s+(\S+)',elapsedTime[0]).group(1))
                else:
                    print("No soln for file %s" % (fileName))
                    if len([x.strip() for x in output if 'Bound:' in x]) == 0:
                        print("Error in file %s" % (fileName))
                    soln = None
                    nodesGenerated_num = None
                    elapsedTime = None
                    noSolnNum = noSolnNum +1

                solns.append(soln)
                nodesGenerated_nums.append(nodesGenerated_num)
                elapsedTimes.append(elapsedTime)

            print("Total of %i files don't have a solution" % (noSolnNum))



        return(solns,nodesGenerated_nums,elapsedTimes)

