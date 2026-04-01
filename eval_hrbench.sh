python eval/eval_hrbench.py \
    --model_name deepeyes \
    --api_url http://localhost:8000/v1 \
    --hrbench_path ./data/HR-Bench \
    --save_path ./results/HR-Bench \
    --eval_model_name deepeyes \
    --num_workers 16