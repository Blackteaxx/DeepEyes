cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/DeepEyes
conda activate deepeyes
export WORLD_SIZE=1
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1

ray stop
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ray start --head --dashboard-host=0.0.0.0 --num-gpus 6 --num-cpus 32

# ------------------------------------------------------------

conda activate deepeyes
export WORLD_SIZE=1
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1
source /opt/rh/devtoolset-8/enable

which gcc
gcc --version
which g++
g++ --version

find /opt/rh/devtoolset-8 -name stdatomic.h 2>/dev/null | head

export CC=$(which gcc)
export CXX=$(which g++)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash examples/agent/final_merged_v1v8_thinklite_3b.sh