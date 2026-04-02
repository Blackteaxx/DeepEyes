vllm serve ../model/DeepEyes-7B \
  --port 8000 \
  --tensor-parallel-size 4 \
  --served-model-name deepeyes \
  --trust-remote-code \
  --limit-mm-per-prompt "image=5" \
  --gpu-memory-utilization 0.6