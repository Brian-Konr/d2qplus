# please run this script in the root directory of the project
export CUDA_VISIBLE_DEVICES=4
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$CONDA_PREFIX/lib/server/libjvm.so"

DATASET=fiqa-5000
python experiments/score_generator.py \
    --input /home/guest/r12922050/GitHub/d2qplus/gen/$DATASET/t5_100q.jsonl \
    --output /home/guest/r12922050/GitHub/d2qplus/gen/$DATASET/scores-t5/t5_100q.jsonl \
    --log /home/guest/r12922050/GitHub/d2qplus/gen/$DATASET/scores-t5/t5_100q.log