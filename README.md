# Multi Type Partially Observable Mean Field Multi-Agent Reinforcement Learning 

Implementation of MTPOMF-Q.

 
## Code structure


- `./examples/`: contains scenarios for Battle Game (also models).

- `battle.py`: contains code for running Battle Game with trained model

- `train_battle.py`: contains code for training Battle Game models

## Compile Ising environment and run

**Requirements**
- `python==3.6.1`
- `gym==0.9.2` (might work with later versions)
- `matplotlib` if you would like to produce Ising model figures

## Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent)

**Steps for compiling**

```shell
cd examples/battle_model
./build.sh
```

**Steps for training models under Battle Game settings**

1. Add python path in your `~/.bashrc` or `~/.zshrc`:

    ```shell
    vim ~/.zshrc
    export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
    source ~/.zshrc
    ```

2. Run training script for training (example):

    ```shell
    python3 train_battle.py --algo pomtmfq --n_round 3001 --max_steps 500 --render --save_every 250

    ```
    
3. To reproduce results of the report:
   ```shell
   python3 battle.py --algo mtmfq --oppo pomtmfq --n_round 100 --idx 2999 2999 --max_steps 500 --render --mtmfqp 0
   ```
   
4. Code was initialy forked from the following [repo](https://github.com/mlii/mfrl.git).
