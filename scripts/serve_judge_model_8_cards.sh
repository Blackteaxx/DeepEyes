vllm serve ../model/Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 \
  --port 18901 \
  --tensor-parallel-size 8 \
  --served-model-name judge \
  --trust-remote-code \
  --disable-log-requests