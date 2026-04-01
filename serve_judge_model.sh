CUDA_VISIBLE_DEVICES=0,1 vllm serve ../model/Qwen/Qwen2.5-32B-Instruct \
  --port 18901 \
  --tensor-parallel-size 2 \
  --served-model-name judge \
  --trust-remote-code \
  --disable-log-requests