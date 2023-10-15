from dataclasses import dataclass, field
from typing import Optional, List

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from HF"}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length"}
    )
    task_name: str = field(
        default=None,
        metadata={"help": "Task name: detector, nlp, llm"}
    )

    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations, in-context learning
    few_shot_type: str = field(
        default='finetune',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    l2_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use L2 loss (only makes a difference in standard FT)."}
    )

    # LoRA arguments: only for BERT-type model
    apply_lora: bool = field(
        default=False,
        metadata={'help': 'use LoRA for fine tuning'}
    )
    lora_alpha: int = field(
        default=None,
        metadata={'help': 'initialization scale for one of the low rank matrices in lora'}
    )
    lora_r: int = field(
        default=None,
        metadata={'help': 'inner rank for lora matrices'}
    )

    # Calibration
    sfc: bool = field(
        default=False,
        metadata={"help": "Whether to use surface form calibration."}
    )

    icl_sfc: bool = field(
        default=False,
        metadata={"help": "Use in-context learning demos in sfc."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    # For prompting
    sfc_prompt: str = field(
        default=None,
        metadata={"help": "SFC prompt"}
    )
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )
    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )
    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )
    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )
    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )
    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )
    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )
    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )
    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )
    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )
    gpt3_demo_separator: str = field(
        default="\n\n\n",
        metadata={"help": "Separator between demonstrations"}
    )
    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )
    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: List[str] = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training or at the end"}
    )

    log_file: str = field(
        default='log'
    )

    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-parameter search) to identify the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    optimizer_variant: str = field(
        default='',
        metadata={'help': 'define variants on optimizer: signgd'}
    )

    trainer: str = field(
        default="standard",
        metadata={"help": "Pick from {standard, kernel, linearhead}"}
    )
    from_linearhead: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize head with the linearhead solution. Works for both normal and kernel trainer."}
    )
    lp_early_stopping: bool = field(
        default=False,
        metadata={
            "help": "When on, increases the tolerance and lowers max_iter in scikit LogisticRegression solver to encourage early stopping."}
    )
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    sweep: bool = field(
        default=False,
        metadata={'help': 'configures the output directories to be informative when running W&B sweep'}
    )

    num_prefix: int = field(
        default=10,
        metadata={"help": "How many prefix tokens to use"}
    )
    no_reparam: bool = field(
        default=False,
        metadata={"help": "No reparameterization trick"}
    )
    prefix_init_by_real_act: bool = field(
        default=False,
        metadata={
            "help": "For no_reparam case, randomly sample words and take their actual key/value pairs as initialization"}
    )
    layer_wise_optim: bool = field(
        default=False,
        metadata={'help': 'Optimize layer-by-layer (only for prefix + ZO)'}
    )

    max_zo_forward_steps: int = field(
        default=0,
        metadata={
            'help': 'Stop at this number of ZO forward steps. The trainer will take whichever is reached first, max_steps or max_zo_forward_steps.'}
    )

    optimize_acc: bool = field(
        default=False,
        metadata={"help": "Maximize accuracy instead of minimizing loss"}
    )

    hf_inference_model: bool = field(
        default=False,
        metadata={
            "help": "loads the HF model in inference mode across many GPUs. incompatible with --zero_order_use_trainer_optim."}
    )
