## Requirements

```bash
pip install -r requirements.txt
```

## Run

Start the vLLM server first.

```bash
bash start_vllm.sh /path/to/ChatGLM3-6B 8000 0
```

Or start it manually:

```bash
conda activate vllm
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 \
    python -m vllm.entrypoints.openai.api_server \
    --model /path/to/ChatGLM3-6B --port 8000
```

## Generate training data

```bash
python generate_dvm_data.py --model /path/to/ChatGLM3-6B --game_count 10 --wr_cons 0.5
```

## Run evaluation

```bash
python run_dvm_werewolf.py --camp villager --game_count 10 --wr_cons 0.5
