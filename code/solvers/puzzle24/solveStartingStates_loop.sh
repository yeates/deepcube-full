ITR_START=$1
ITR_END=$2

for (( ITR=$ITR_START; ITR<$ITR_END; ITR++ ))
do
	CMD="sh scripts/runSGE_cpu.sh 1-9 1 \"python scripts/solveStartingStates.py --input testData/puzzle24/states.pkl --env puzzle24 --methods optimal --startIdx $ITR --endIdx $(($ITR + 1))\""
	echo $CMD
	eval $CMD
done
