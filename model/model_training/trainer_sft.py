#!/usr/bin/env python3
import argparse
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import datasets
import torch

import numpy as np
from model_training.custom_datasets.formatting import DatasetEntry
from model_training import print_rank_0
from model_training.custom_datasets.dialogue_collator import DialogueDataCollator
from model_training.efficiency_utils import fuse_gelu
#from model_training.models.patching import RopePatch
#from model_training.models.peft_modeling import peft_model
from model_training.utils.utils import (
    PerDatasetSampler,
    _strtobool,
    get_dataset,
    get_loss,
    get_metrics,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
from transformers import EarlyStoppingCallback
from transformers.utils import is_datasets_available


def compute_metrics(eval_pred, preprocess_fns, metrics):
    out = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        out = dict(**out, **metric.compute(predictions=preds, references=labels))
    
    return out


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class EarlyStopEvalLossCallback(TrainerCallback):


    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    """
    If evaluation loss from oasst data stops improving past a 2
    """
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_oasst= "eval_oasst_export_loss"
        metric_qa="eval_finnish_instruction_qa_loss"
        metrics_keys = list(metrics.keys())
        metric_to_check = metrics_keys[0]
        logging.debug(f"Metric that is checked on evaluate: {metric_to_check}")
        if metric_to_check is None:
            print_rank_0(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping \
                is disabled")
            return

        self.check_metric_value(args, state, control, metric_to_check)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        sampler: torch.utils.data.sampler.Sampler = None,
        loss_function: str = "CrossEntropyLoss",
        poly_eps: float = 1.0,
        train_collate_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.train_collate_fn = train_collate_fn
        # By default CrossEntropyLoss ignores padding_index -100, but just in case use our own loss_fct
        self.loss_fct = get_loss(loss_function, poly_eps)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_outputs=False):
        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)

        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        logits = outputs.get("logits")

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return loss, logits, targets, labels_mask

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, logits, labels, labels_mask = self._compute_loss(model, inputs)
            labels[~labels_mask.bool()] = -100  # padding_index

        loss = loss.mean().detach()

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")
        

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader


def argument_parsing(notebook: bool = False, notebook_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./trainer_sft.py --configs galactica-125m webgpt_dataset_only per_digit_tokens

        to run the galactica-125m model, using the webgpt dataset only (as opposed to all
        the datasets listed in defaults in config.yaml) and treat each digit as a separate token.
    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--push_to_hub",type=bool,default=False)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from last saved checkpoint")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--show_dataset_stats", action="store_true", help="Show dataset stats", default=False)
    parser.add_argument("--report_to",type=str,help="The list of integrations to report the results and logs to", default='none')
    parser.add_argument("--debug",action="store_true")
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print_rank_0(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    conf["report_to"] = args.report_to
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = int(os.environ['WORLD_SIZE'])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)


def tokenizer_sanity_check(tokenizer):
    print_rank_0("Tokenizer sanity check:")
    print_rank_0(f"Type: {type(tokenizer).__name__}")

    print_rank_0(f"special_tokens_map: {tokenizer.special_tokens_map}")

    print_rank_0(f"bos_token={tokenizer.bos_token} bos_token_id={tokenizer.bos_token_id}")
    print_rank_0(f"eos_token={tokenizer.eos_token} eos_token_id={tokenizer.eos_token_id}")

    from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, create_dataset_entry_qa

    ds_entry = create_dataset_entry_qa(
        mode="sft", questions=["Q1", "Q2"], answers=["A1", "A2"], lang="en", context="ctx"
    )
    in_text = ds_entry.get_formatted(
        tokenizer.eos_token,
        use_system_tag=True,
        system_property_dropout=0,
        system_add_length=True,
    )
    in_text = "".join(in_text)

    prompter_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Question"])
    assistant_token_id = tokenizer.convert_tokens_to_ids(QA_SPECIAL_TOKENS["Answer"])
    print_rank_0(f"{prompter_token_id=}, {assistant_token_id=}")

    tr = tokenizer(in_text, max_length=1024, pad_to_max_length=False, truncation=True)

    message_indices = []
    i = -1
    for id in tr.input_ids:
        if id in (prompter_token_id, assistant_token_id):
            i += 1
        message_indices.append(i)

    print_rank_0(f"encoding result:, {tr}")
    for i, xs in enumerate(tr.input_ids):
        decoded = tokenizer.decode(xs)
        print_rank_0(f'{i}: {xs} -> "{decoded}"')

    print_rank_0(f"message_indices: {message_indices}")


def main():
    training_conf = argument_parsing()

    #if not training_conf.deepspeed or training_conf.local_rank == 0:
        #print_rank_0(f"trainig_conf = {training_conf}")

    output_dir = (
        training_conf.output_dir
        if training_conf.output_dir
        else f"{training_conf.model_name}-{training_conf.log_dir}-finetuned"
    )

    optimizer = OptimizerNames.ADAMW_BNB if training_conf.quantization else OptimizerNames.ADAMW_HF
    
    # needs to happen before model loading in case of stage 3 training
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed_config if training_conf.deepspeed else None,
        optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        half_precision_backend='auto',#For amd either cpu_amp or apex is allowed
        local_rank=os.environ['SLURM_LOCALID'],
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        eval_accumulation_steps=training_conf.eval_accumulation_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to=training_conf.report_to,
        dataloader_num_workers=0,
        log_level=training_conf.log_level,
        log_on_each_node=False,
        #full_determinism=True
    )

    json_args = args.to_json_string()
    """with open(output_dir/training_arguments.json,'w') as f:
        f.write(json_args)"""

    init_rng(training_conf)

    tokenizer = get_tokenizer(training_conf)

    if not training_conf.deepspeed or training_conf.local_rank == 0:
        tokenizer_sanity_check(tokenizer)

    train_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        random_offset_probability=training_conf.random_offset_probability,
        label_masking=training_conf.label_masking,
        samples_mixing=training_conf.samples_mixing,
        use_system_prefix=training_conf.use_system_prefix,
        system_prefix=training_conf.system_prefix,
        use_system_tag=training_conf.use_system_tag,
        system_property_dropout=training_conf.system_property_dropout,
        system_add_length=training_conf.system_add_length,
    )

    if training_conf.val_max_length is None:
        training_conf.val_max_length = training_conf.max_length

    eval_collate_fn = DialogueDataCollator(
        tokenizer,
        max_length=training_conf.val_max_length,
        random_offset_probability=training_conf.random_offset_probability,
        label_masking=training_conf.label_masking,
        samples_mixing=False,
        use_system_prefix=training_conf.use_system_prefix,
        system_prefix=training_conf.system_prefix,
        use_system_tag=training_conf.use_system_tag,
        system_property_dropout=training_conf.system_property_dropout,
        system_add_length=training_conf.system_add_length,
    )

    train, evals = get_dataset(training_conf)
    show_dataset_stats = (training_conf.verbose or training_conf.show_dataset_stats) and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    )
    if show_dataset_stats:
        print_rank_0("Training dataset sizes (before sampling):")
        total = len(train)
        for d in train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print_rank_0(f"{name}: {len(d)} ({len(d) / total:.2%})")

            # ensure that all entries can be formatted
            # for x in d:
            #     if isinstance(x, DatasetEntry):
            #         x.get_formatted("sft", "<eos>")

        print_rank_0(f"\nTotal train: {total}")
        print_rank_0("-" * 80)
        print_rank_0("Evaluation set sizes:")
        total_eval = sum(len(x) for x in evals.values())
        for k, d in evals.items():
            print_rank_0(f"{k}: {len(d)} ({len(d) / total_eval:.2%})")
        print_rank_0(f"\nTotal eval: {total_eval}")
        print_rank_0("-" * 80)

    if training_conf.use_custom_sampler:
        samples_length = None
        if training_conf.sort_by_length:
            samples_length = list(
                map(
                    lambda x: train_collate_fn.process_one(x, return_length=True),
                    tqdm(train, desc="Calculating lengths per sample"),
                )
            )

        sampler = PerDatasetSampler.build_sampler_from_config(
            training_conf,
            train.datasets,
            rank=training_conf.local_rank,
            world_size=training_conf.world_size,
            samples_length=samples_length,
            verbose=show_dataset_stats,
        )
    else:
        sampler = None

    metrics, preprocess_fns = get_metrics(training_conf, tokenizer)
    model = get_model(training_conf, tokenizer)

#    superhot = RopePatch.from_config(training_conf) if training_conf.superhot else None
#    if superhot:
#        superhot.patch(model)

#    print(f"rope_scaling: {model.config.rope_scaling}")
#    print(f"max_position_embeddings: {model.config.max_position_embeddings}")

    #if training_conf.peft_model:
    #    print("Using PEFT model")
    #    model = peft_model(model, training_conf)

    if training_conf.quantization:
        import bitsandbytes  # This is noisy, so delay importing until after argument parsing so it doesn't make --help noisy

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, "weight", {"optim_bits": 32}
                )

    if training_conf.fuse_gelu:
        model = fuse_gelu(model)

    if not training_conf.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if training_conf.log_wandb and (not training_conf.deepspeed or training_conf.local_rank == 0):
        import wandb

        wandb_name = training_conf.model_name.replace(os.getenv("HOME", "/home/ubuntu"), "")
        wandb.init(
            project="supervised-finetuning",
            entity=training_conf.wandb_entity,
            resume=training_conf.resume_from_checkpoint,
            name=f"{wandb_name}-{training_conf.log_dir}-finetuned",
            config=training_conf,
        )
        wandb.config["_max_length"] = training_conf.max_length
        wandb.config["_val_max_length"] = training_conf.val_max_length

    trainer = SFTTrainer(
        model=model,
        args=args,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=training_conf.loss_fn,
        poly_eps=training_conf.poly_eps,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_fns),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    #trainer.add_callback(EarlyStopEvalLossCallback(3, 0.0))
    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
