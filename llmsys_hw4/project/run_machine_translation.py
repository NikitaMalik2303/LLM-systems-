from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
import argparse
from distutils.util import strtobool

from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps


def get_dataset(dataset_name, model_max_length):
    """
    Obtain IWSLT (de-en) dataset.
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]             # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Trains a tokenizer on the provided dataset examples and saves the tokenizer configuration.

    Parameters:
    - examples: The dataset examples used for training the tokenizer.
    - vocab_size: The desired vocabulary size for the tokenizer.
    - src_key: The key used to access the source text within the dataset examples.
    - tgt_key: The key used to access the target text within the dataset examples.
    - workdir: The directory where the tokenizer should be saved.

    Returns:
    - tokenizer: The trained tokenizer with special tokens,
        e.g., ("<eos_de>", "<eos_en>", "<pad>") if src_key and tgt_key are "de" and "en", respectively.
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        examples (list): List of examples to process
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding texts
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors

    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequences of shape (batch_size, model_max_length-1)
            - labels: Target sequences of shape (batch_size, model_max_length-1)
            - label_token_weights: Weight mask for loss computation of shape (batch_size, model_max_length-1)
            
    Note:
        input_ids format: <de_tokens> + <de_eos> + <en_tokens> + <en_eos> + <pad>
        labels: Next tokens to predict (shifted by 1)
        label_token_weights: 0 for source tokens, 1 for target tokens
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    The MLE loss for a batch.

    Parameters:
    - batch: The result of collate_fn, a dict with "input_ids", "labels", and "label_token_weights".
    - model: The model to be trained.

    Returns:
    - A scalar loss value for this batch, averaged across all target tokens.
    """

    idx = batch['input_ids']
    idx.requires_grad_(True)
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    targets.requires_grad_(True)
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        
        batch = collate_fn(examples=examples[i:i + batch_size])
        
        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        # print(f"Forward: {t1 - t0}")
        # print(f"Backward: {t2 - t1}")
        # print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)


def parse_args():
    def str2bool(x):
        return bool(strtobool(x))
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-fused-kernel', type=str2bool, default=False)
    return parser.parse_args()


def main(dataset_name='bbaaaa/iwslt14-de-en-preprocess',
         model_max_length=40,
         n_epochs=1,
         batch_size=128,
         learning_rate=0.02,
         samples_per_epoch=20000,
         n_vocab=10000,
         n_embd=256,
         seed=11111, 
         use_fused_kernel=False):
    #args = parse_args()
             
    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab'     : n_vocab,  # vocab_size
        'n_embd'      : n_embd,   # n_embed
        'n_head'      : 8,    # n_head
        'n_positions' : model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout'   : 0.1,  # x_pdrop
        'ln_eps'      : 1e-5, # layer_norm_epsilon
        'backend'     : backend,
        'use_fused_kernel': use_fused_kernel
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)
    
    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)


#if __name__ == '__main__':
#    fire.Fire(main)

main()
