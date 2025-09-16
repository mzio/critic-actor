# Critic-Actor (Cria)

Guiding your test-time compute with RL (your critic is an actor now).

## Setup

We use `uv` for package management. You can install it from [here](https://docs.astral.sh/uv/getting-started/installation/).

### Environment variables and API keys

To manage model API keys, create a `.env` file in this project's root directory, e.g.,:

```bash
# to store datasets, checkpoints, etc.
ROOT_DIR="/home/ubuntu/data" 

# Model API keys
OPENAI_API_KEY="sk-proj-abc123..."
```

If not already there, add this to your `.gitignore` so it doesn't drift into our repo.

## Sample Commands

GPQA with GPT-5-mini

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
uv run python main.py \
--env_config gpqa/gpqa \
--llm_config api/oai_gpt5_mini \
--model_config hf/llama3_1_8b_inst \
--critic_actor_config cria_attn_pool_cos \
--trainer_config critic_actor \
--replay_buffer_config cria \
--max_new_tokens 4096 \
--samples_per_prompt 8 \
--prompts_per_update 8 \
--update_batch_size 32 \
--gradient_accumulation_steps 32 \
--lr 1e-4 \
--weight_decay 0 \
--max_turns 10 \
--replicate 0 --seed 0 --verbose
```

GPQA with GPT-4.1-mini

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
uv run python main.py \
--env_config gpqa/gpqa \
--llm_config api/oai_gpt4_1_mini \
--model_config hf/llama3_1_8b_inst \
--critic_actor_config cria_attn_pool_cos \
--trainer_config critic_actor \
--replay_buffer_config cria \
--max_new_tokens 4096 \
--samples_per_prompt 8 \
--prompts_per_update 8 \
--update_batch_size 32 \
--gradient_accumulation_steps 32 \
--lr 1e-4 \
--weight_decay 0 \
--max_turns 10 \
--replicate 0 --seed 0 --verbose
```

BrowseComp with GPT-4.1-mini

```bash

```