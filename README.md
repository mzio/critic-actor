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

