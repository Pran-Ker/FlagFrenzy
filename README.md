# War Games Track

Can we build AI systems to assist with the education of officers in war games? How can we accelerate strategic learning with a specific war game? Can we build or train an AI system to compete and win war games efficiently? Hackers in this track will need to reference a copy of an existing war game. For AI training a Linux environment is provided in this repository. For desktop/laptop coaching and play, a windows and mac build are provided [here](https://drive.google.com/drive/folders/142WEUmis1x9bcuGAznaUcu6n8L8kfWGb?usp=drive_link)

*Example Projects*

- Develop an intelligent assistant for war games, capable of guiding players through complex operations, offering context and strategy suggestions at key moments, and providing feedback

- Build a game analyzer to thoroughly commend good actions, evaluate and correct mistakes, or otherwise provide post-game analysis

- Build (via agentic frameworks and LLMs) or train (via fine tune, SFT, or RL) an AI system capable of playing and winning war games effectively, explaining each choice and maintaining maximum observability

## HOW TO PLAY 

https://www.youtube.com/watch?v=s2NVnsqlAoE

## Wargaming Track AI Training

For those wanting to train an AI.

FlagFrenzyEnv is a custom Gymnasium-compatible air-combat simulation environment designed for reinforcement learning research. The agent commands air units from the Blue Team to complete strategic objectives in a continuous, fully observable battlefield.

Full environment documentation, action and observation space specifications, and other relevant developer details are available on [Notion](https://www.notion.so/FlagFrenzyEnv-173a6b5d68fa80e3a73afcf095620bbf?pvs=4).

---

### Setup Instructions

Python version requirement: **3.9 only** (must match simulation bindings)

We recommend using a Python virtual environment:

```bash
python3.9 -m venv venv
source venv/bin/activate         # On Linux/macOS
venv\Scripts\activate.bat        # On Windows
pip install -r requirements.txt
```

If you need Python 3.9, use [pyenv](https://github.com/pyenv/pyenv) for easy version management.

This project is compatible with Linux, Windows, and macOS.

---

### Running the Environment

#### Verify your setup

To confirm that the environment and simulation bindings are working:

```bash
python test_env.py
```

This will initialize the FlagFrenzyEnv environment, sample random actions, and execute them to verify that everything is functioning correctly.

#### Train with PPO

Run the included PPO training script:
```bash
python train.py
```

- Uses Ray RLlib PPO with a custom hybrid action model
- Trains for approximately 19 million timesteps (450 iterations)
- Logs training results to ~/ray_results/flag_frenzy_ppo/

You are encouraged to modify the training configuration and model setup to experiment with:
- Learning rates
- Batch sizes
- Model architecture
- Action distribution
- Reward functions

### Issues and Contributions

If you find a bug or have a feature request, please open an issue. Contributions are welcome through pull requests.

#### Author
Sanjna Ravichandar
sanjna@codemetal.ai
