<a href="https://github-com.translate.goog/LAION-AI/Open-Assistant/blob/main/model/README.md?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp">![Translate](https://img.shields.io/badge/Translate-blue)</a>

## Reproduction directions

Here are some minimal commands to tun to whole pipeline on the collected data.

**make sure python >= 3.10, otherwise, you would meet the
[[issue]](https://github.com/tiangolo/typer/issues/371#issuecomment-1288987924)**

1. Setup for lumi
'''bash
./get_started.sh

### SFT Training

2. Start with the SFT training.

```bash
cd model_training
sbatch slurm/sft_train.sh

To change the model used, i.e. larger pythia version create a new config in
`model_training/configs/config.yaml` or set the flag `--model_name` to
`EleutherAI/pythia-{size}-deduped`. Larger models will probably need to also
adjust the `--learning_rate` and `--per_device_train_batch_size` flags.

3. Get SFT trained model

```bash
# choose a specific checkpoint
export SFT_MODEL=$MODEL_PATH/sft_model/<checkpoint-X>

# or get latest checkpoint
export SFT_MODEL=$MODEL_PATH/sft_model/$(ls -t $MODEL_PATH/sft_model/ | head -n 1)
```

### RM Training

4. Train the reward model

```bash
sbatch slurm/rm_train.sh
```

5. Get RM trained model

```bash
# choose a specific checkpoint
export REWARD_MODEL=$MODEL_PATH/reward_model/<checkpoint-X>

# or get latest checkpoint
export REWARD_MODEL=$MODEL_PATH/reward_model/$(ls -t $MODEL_PATH/reward_model/ | head -n 1)
```

### RL Training

7. Train the RL agent

```bash
cd model_training
python trainer_rl.py --configs defaults_rlhf --cache_dir $DATA_PATH --rank_model $REWARD_MODEL --sft_model $SFT_MODEL --output_dir $MODEL_PATH/rl_model
```

# Message and Token Format

See the `MESSAGE_AND_TOKEN_FORMAT.md` file for information about the pattern we
are using.
