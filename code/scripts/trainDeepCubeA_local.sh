CMD_BASE=$1
BATCH_SIZE=$2
DIR=$3
SCRAMB_MAX=$4
NUM_STEPS=$5
ENV=$6
NNET_NAME=$7
EPS=$8

NUM_BATCHES_PER_JOB=5000
NNET_EXPORT_NUM=0
NNET_TRAIN_NUM=1

mkdir $DIR

GEN_DATA_CMD_BASE="python scripts/makeLabeledSet.py --env $ENV --num_states $BATCH_SIZE --scramb_max $SCRAMB_MAX --search_depth 1 --dir $DIR --num_itrs $NUM_BATCHES_PER_JOB --num_steps $NUM_STEPS"

echo "BASE CMD: $CMD_BASE\n"
echo "BASE GEN DATA CMD: $GEN_DATA_CMD_BASE\n"

ITR=1
while true
do
	eval "rm -f $DIR/*.pkl"
    eval "rm -f $DIR/*_done"

	### Prepare training command 
	CMD=$CMD_BASE
	
	CMD="$CMD --scramb_max $SCRAMB_MAX --env $ENV --batch_size $BATCH_SIZE --labeled_data $DIR --delete_labeled --nnet_name $NNET_NAME --model_num $NNET_TRAIN_NUM --eps $EPS"

	### Train network in background
	echo "RUNNING CMD: $CMD"
	eval "$CMD &"
	PID_TRAIN=$(echo $!)
	PID_DATA_GEN=""
	echo "PID IS: $PID_TRAIN\n"

	### Generate data
	BASE_IDX=0
	while ps -p $PID_TRAIN > /dev/null
	do
		if [ "$PID_DATA_GEN" = "" ]; then
			NUM_FILES=$(eval "ls $DIR" | wc | awk '{print $1}')
			if [ $NUM_FILES -lt "1000" ]; then
				### RUN DATA GENERATION
				echo "RUNNING DATA GEN\n"

				if [ -d savedModels/$NNET_NAME/$NNET_EXPORT_NUM/ ]; then
					GEN_DATA_CMD="$GEN_DATA_CMD_BASE --model_loc savedModels/$NNET_NAME/$NNET_EXPORT_NUM/ --base_idx $BASE_IDX"
				else
					GEN_DATA_CMD="$GEN_DATA_CMD_BASE --model_loc \"\" --base_idx $BASE_IDX"
				fi

				eval "$GEN_DATA_CMD &"

				PID_DATA_GEN=$(echo $!)

				BASE_IDX=$(($BASE_IDX + $NUM_BATCHES_PER_JOB))
			fi
		elif ! ps -p $PID_DATA_GEN > /dev/null; then
			PID_DATA_GEN=""
		fi
		sleep 5
	done

	echo "TRAINING DONE\n"
	
	if ps -p $PID_DATA_GEN > /dev/null; then
		echo "KILLING DATA GENERATING JOB"
		kill $PID_DATA_GEN
		PID_DATA_GEN=""
	fi

	ITR=$(($ITR + 1))
	
	eval "rm -f savedModels/$NNET_NAME/$NNET_TRAIN_NUM/events.out.tfevents*"
	eval "rm -rf savedModels/$NNET_NAME/$NNET_EXPORT_NUM/"
	eval "cp -r savedModels/$NNET_NAME/$NNET_TRAIN_NUM savedModels/$NNET_NAME/$NNET_EXPORT_NUM"
done
