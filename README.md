# DVM: Werewolf Agent

Implementation of DVM for controllable LLM agents in Werewolf.

## Requirements

```bash
pip install -r requirements.txt
```

## Model

The paper uses ChatGLM3-6B as the base model for Predictor and Discussor. Update `configs/dvm_chatglm3-6b.yaml` if your path differs.

## Run

Start vllm server first:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /path/to/ChatGLM3-6B --port 8000
```

Run evaluation:

```bash
python run_dvm_werewolf.py --camp villager --game_count 10 --wr_cons 0.5
```

For different win rate constraints:

```bash
python run_dvm_werewolf.py --camp villager --game_count 30 --wr_cons 0.3
python run_dvm_werewolf.py --camp villager --game_count 30 --wr_cons 0.7
```
