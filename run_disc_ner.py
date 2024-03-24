import os
import re
import sys
import json
import yaml
import math
import random
import argparse
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from copy import deepcopy
from itertools import chain, product
from collections import defaultdict, Counter

use_wandb = False
if os.environ.get("WANDB_ENABLE", False):
    try:
        import wandb
        use_wandb = True
        """
        - https://github.com/wandb/examples
        - https://docs.wandb.ai/guides/sweeps
        """
    except ImportError:
        use_wandb = False

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import (
    BertConfig, BertPreTrainedModel, BertModel, BertTokenizerFast,
    RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaTokenizerFast,
    PreTrainedModel, AdamW,
)
from transformers.file_utils import ModelOutput

from torchblocks.callback import ProgressBar
# from torchblocks.data.dataset import DatasetBase
from torchblocks.data.process_base import ProcessBase
from torchblocks.metrics.sequence_labeling.scheme import get_scheme
from torchblocks.metrics.sequence_labeling.precision_recall_fscore import _precision_recall_fscore_support
from torchblocks.metrics.sequence_labeling.seqTag_score import SequenceLabelingScore
from torchblocks.layers.layer_norm import ConditionalLayerNorm
from torchblocks.losses.focal_loss import FocalLoss
from torchblocks.losses.label_smoothing import LabelSmoothingCE
from torchblocks.core import TrainerBase
from torchblocks.utils.options import Argparser
from torchblocks.utils.logger import Logger
from torchblocks.utils.device import prepare_device
from torchblocks.utils.paths import check_dir, load_pickle, check_file, is_file
from torchblocks.utils.paths import find_all_checkpoints
from torchblocks.utils.seed import seed_everything


IGNORE_INDEX = -100
Span = NewType("Span", Tuple[int, int, str])
Entity = NewType("Entity", List[Span])

import logging
logger = logging.getLogger(__name__)

def default_data_collator(features: List[Dict[str, torch.Tensor]],
                          dynamic_batch=False,
                          dynamic_keys=[]) -> Dict[
    str, Any]:
    batch = {}
    first = features[0]
    max_input_length = first['input_ids'].size(0)
    if dynamic_batch:
        max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            bv = torch.stack([f[k] for f in features]) if isinstance(v, torch.Tensor) else torch.tensor(
                [f[k] for f in features])
            batch[k] = bv
    if dynamic_batch:
        for k in dynamic_keys:
            if k not in batch: continue
            if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
    return batch


class DatasetBase(torch.utils.data.Dataset):
    keys_to_truncate_on_dynamic_batch = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 max_examples: int = None,
                 use_cache: bool = False,
                 collate_dynamic: bool = True,
                 cached_features_file: str = None,
                 overwrite_cache: bool = False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        if not is_file(data_name):
            file_path = os.path.join(data_dir, data_name)
            check_file(file_path)
        self.examples = self.create_examples(self.read_data(file_path), data_type)
        if max_examples is not None: self.examples = self.examples[: max_examples]
        self.process_piplines = process_piplines if isinstance(process_piplines, list) else [process_piplines]
        self.num_examples = len(self.examples)
        self.num_labels = len(self.get_labels())
        self.use_cache = use_cache
        self.collate_dynamic = collate_dynamic
        self.cached_features_file = cached_features_file
        if self.cached_features_file is None: self.cached_features_file = ""
        self.overwrite_cache = overwrite_cache
        if self.use_cache:
            self.create_cache()

    def create_cache(self):
        self.cached_features_file = os.path.join(self.data_dir, self.cached_features_file)
        if is_file(self.cached_features_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file %s", self.cached_features_file)
            self.features = torch.load(self.cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {self.data_dir}")
            self.features = [
                self.process_example(example) for example in
                tqdm(self.examples, total=self.num_examples, desc="Converting examples to features...")]
            # FIXED: is_file必须在文件存在情况下才返回True
            # if is_file(self.cached_features_file):
            if not os.path.isdir(self.cached_features_file):
                logger.info("Saving features to cached file %s", self.cached_features_file)
                torch.save(self.features, self.cached_features_file)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.use_cache:
            feature = self.features[index]
        else:
            feature = self.process_example(self.examples[index])
        return feature

    def __len__(self):
        return self.num_examples

    @classmethod
    def get_labels(self) -> List[str]:
        raise NotImplementedError('Method [get_labels] should be implemented.')

    @classmethod
    def label2id(cls):
        return {label: i for i, label in enumerate(cls.get_labels())}

    @classmethod
    def id2label(cls):
        return {i: label for i, label in enumerate(cls.get_labels())}

    def read_data(self, input_file: str) -> Any:
        raise NotImplementedError('Method [read_data] should be implemented.')

    def create_examples(self, data: Any, set_type: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError('Method [create_examples] should be implemented.')

    def process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        for proc in self.process_piplines:
            if proc is None: continue
            example = proc(example)
        return example

    def collate_fn(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return default_data_collator(features,
                                     dynamic_batch=self.collate_dynamic,
                                     dynamic_keys=self.keys_to_truncate_on_dynamic_batch)

class SpanClassificationDataset(DatasetBase):
    """
    Attributes
    ----------
    
        examples: 
        !!! WARN: word-level

    """
    keys_to_truncate_on_dynamic_batch = [
        "input_ids", "attention_mask", "token_type_ids", "conditional_ids", "syntactic_upos_ids",
    ]

    def __init__(
        self,
        data_name,
        data_dir,
        data_type,
        process_piplines: List[Callable],
        context_size: int = 0,
        max_examples: int = None,
        labels: List[str] = None,
        use_cache: bool = False,
        collate_dynamic: bool = True,
        cached_features_file: str = None,
        overwrite_cache: bool = False
    ) -> None:
        
        self.labels = labels
        self.context_size = max(0, context_size)
        super().__init__(data_name, data_dir, data_type, process_piplines, max_examples,
            use_cache,collate_dynamic, cached_features_file, overwrite_cache)

        # add context
        if self.context_size > 0:
            examples = []
            for i, example in tqdm(enumerate(self.examples), total=len(self.examples), 
                                    desc=f"Adding Context({self.context_size})..."):
                examples.append(self.set_example_context(example, i))
            self.examples = examples

    def set_example_type(self, example):
        is_overlap = False
        is_discontinuous = False

        if example["entities"] is not None: 

            entities = []
            for entity in example["entities"]:
                entities.append(set())
                if len(entity) > 1:
                    is_discontinuous = True
                for start, end, _, _ in entity:
                    for pos in range(start, end):
                        entities[-1].add(pos)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity_a, entity_b = entities[i], entities[j]
                    if len(entity_a.intersection(entity_b)) > 0:
                        is_overlap = True

        example["is_overlap"] = is_overlap
        example["is_discontinuous"] = is_discontinuous
        return example
    
    def set_example_context(self, example, example_no):
        example = deepcopy(example)
        # left context
        left_context_size = 0
        context_no = example_no
        while True:
            context_no -= 1
            if context_no < 0:
                break

            context_size = max(0, self.context_size - left_context_size)
            context_text = self.examples[context_no]["text"][- context_size: ]
            left_context_size += len(context_text)
            example["text"] = context_text + example["text"]
            if left_context_size >= self.context_size:
                break

        # right context
        right_context_size = 0
        context_no = example_no
        while True:
            context_no += 1
            if context_no >= len(self.examples):
                break

            context_size = max(0, self.context_size - right_context_size)
            context_text = self.examples[context_no]["text"][: context_size]
            right_context_size += len(context_text)
            example["text"] = example["text"] + context_text
            if right_context_size >= self.context_size:
                break

        example["sent_start"] = left_context_size
        example["sent_end"] = len(example["text"]) - right_context_size
        if example["entities"] is not None:
            for entity in example["entities"]:
                for i, (start, end, label, string) in enumerate(entity):
                    entity[i] = (start + left_context_size, end + left_context_size, label, string)
                
        return example

    def collate_fn(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        first = features[0]
        max_input_length = None
        if self.collate_dynamic:
            max_input_length = max([torch.sum(f["attention_mask"]) for f in features])
        for k in first.keys():
            bv = None
            if k in ["input_ids", "attention_mask", "token_type_ids", "conditional_ids", "syntactic_upos_ids"]:
                bv = torch.stack([f[k] for f in features], dim=0)                   # (batch_size, sequence_length)
            elif k in ["spans", "spans_mask", "labels"]:
                if first[k] is not None:
                    bv = pad_sequence([f[k] for f in features], batch_first=True)   # (batch_size, num_spans, *)
            elif k in []:
                bv = torch.stack([f[k] for f in features])                          # (batch_size,)
            elif k in []:
                bv = [f[k] for f in features]                                       # (batch_size,)
            else:
                continue
            batch[k] = bv
        if self.collate_dynamic:
            for k in self.keys_to_truncate_on_dynamic_batch:
                if k in batch:
                    if batch[k].dim() >= 2: batch[k] = batch[k][:, : max_input_length]
        if batch["labels"] is None: batch.pop("labels")
        return batch

class CadecDataset(SpanClassificationDataset):
    """
    File Format: 
        ``` txt
        Remaining issue is chest spasm - - have been through tests ruling out heart , GERD , pneumonia , etc Spasms persist although less and less .
        3,4 ADR|20,20 ADR

        Last 48 hours , epsiodes seemed to increase again but may be attributable to some combination of muscle weakness and heavy non - stop rains we ' ve enjoyed for several days .
        17,18 ADR
        ```
    """

    @classmethod
    def get_labels(cls) -> List[str]:
        return ["O", "II-N"] + [f"{prefix}-{label}" for label in (
            "ADR",
        ) for prefix in ("BI", "BE", "II", "IE")]

    def read_data(self, input_file: str) -> Any:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
        return lines

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        texts = data[0::3]; annos = data[1::3]
        assert len(texts) == len(annos)

        examples = []
        for i, (text, anno) in enumerate(zip(texts, annos)):
            guid = f"{data_type}-{i}"
            words = text.split()    # based on whitespace
            entities = None
            if data_type != "test":
                entities = []
                if anno != "":
                    for entity in anno.split("|"):
                        entities.append([])
                        positions, label = entity.split(" ")
                        positions = list(map(int, positions.split(",")))
                        num_spans = len(positions) // 2
                        for j in range(num_spans):
                            start, end = positions[j * 2], positions[j * 2 + 1] + 1     # 左闭右开
                            entities[-1].append((start, end, label, words[start: end]))
            examples.append(self.set_example_type(dict(guid=guid, text=words, entities=entities, sent_start=0, sent_end=len(words))))
        return examples

class ShAReDataset(SpanClassificationDataset):

    @classmethod
    def get_labels(cls) -> List[str]:
        return ["O", "II-N"] + [f"{prefix}-{label}" for label in (
            "Disorder",
        ) for prefix in ("BI", "BE", "II", "IE")]

    def read_data(self, input_file: str) -> Any:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
        return lines

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        texts = data[0::3]; annos = data[1::3]
        assert len(texts) == len(annos)

        examples = []
        for i, (text, anno) in enumerate(zip(texts, annos)):
            guid = f"{data_type}-{i}"
            words = text.split()    # based on whitespace
            entities = None
            if data_type != "test":
                entities = []
                if anno != "":
                    for entity in anno.split("|"):
                        entities.append([])
                        positions, label = entity.split(" ")
                        positions = list(map(int, positions.split(",")))
                        num_spans = len(positions) // 2
                        for j in range(num_spans):
                            start, end = positions[j * 2], positions[j * 2 + 1]
                            if start > end:
                                print(text)
                                print(anno)
                                print(f"({start}, {end}) -> ({end}, {start})")
                                start, end = end, start
                            end += 1    # 左闭右开
                            entities[-1].append((start, end, label, words[start: end]))
            examples.append(self.set_example_type(dict(guid=guid, text=words, entities=entities, sent_start=0, sent_end=len(words))))
        return examples


class LevelConvertorBase:

    do_convert_check = False

    def _convert(self, raw: str) -> Tuple[List[str], List[List[int]]]:
        raise NotImplementedError
    
    def _forward(self, tokens: List[str]) -> str:
        raise NotImplementedError

    def _backward(self, raw: str) -> List[str]:
        return self._convert(raw)[0]

    def forward(self, raw, raw_level_entities):
        tokens, offset_mapping = self._convert(raw)
        raw2token_map = dict()
        for i, (raw_start, raw_end) in enumerate(offset_mapping):
            for j in range(raw_start, raw_end):
                raw2token_map[j] = i
        
        if raw_level_entities is not None:
            token_level_entities = []
            for raw_level_entity in raw_level_entities:
                token_level_entity= []
                for raw_level_start, raw_level_end, label, span_raws in raw_level_entity:

                    try:
                        token_level_start = raw2token_map[raw_level_start]
                        token_level_end = raw2token_map[raw_level_end - 1] + 1
                    except KeyError as e:
                        import pdb; pdb.set_trace()
                    
                    span_tokens = tokens[token_level_start: token_level_end]
                    token_level_string = self._forward(span_tokens)
                    raw_level_string = raw[raw_level_start: raw_level_end]
                    try:
                        assert token_level_string == raw_level_string
                    except AssertionError as e:
                        if self.do_convert_check:
                            print(e)
                            import pdb; pdb.set_trace()

                    token_level_entity.append((token_level_start, token_level_end, label, span_tokens))
                token_level_entities.append(token_level_entity)
        else:
            token_level_entities = None

        return tokens, token_level_entities

    def backward(self, tokens, token_level_entities):
        chars = self._forward(tokens)
        tokens, offset_mapping = self._convert(chars)
        
        if token_level_entities is not None:
            char_level_entities = []
            token2char_map = dict(enumerate(offset_mapping))
            for token_level_entity in token_level_entities:
                char_level_entity = []
                for token_level_start, token_level_end, label, span_tokens in token_level_entity:
                    char_level_start, char_level_end = float('inf'), float('-inf')
                    for i in range(token_level_start, token_level_end):
                        start, end = token2char_map[i]
                        char_level_start = min(char_level_start, start)
                        char_level_end   = max(char_level_end,   end  )
                    
                    span_tokens = tokens[token_level_start: token_level_end]
                    token_level_string = self._forward(span_tokens)
                    char_level_string = chars[char_level_start: char_level_end]

                    try:
                        assert token_level_string == char_level_string
                    except AssertionError as e:
                        if self.do_convert_check:
                            print(e)
                            import pdb; pdb.set_trace()

                    char_level_entity.append((char_level_start, char_level_end, label, char_level_string))
                char_level_entities.append(char_level_entity)
        else:
            char_level_entities = None

        return chars, char_level_entities


class LevelConvertorWhitespace(LevelConvertorBase):
    """
    Example:
        >>> text = "Unified Named Entity  Recognition as Word-Word Relation Classification"
        >>> entities = [
        >>>     [(8, 8 + len("Named Entity  Recognition"), "x", "Named Entity  Recognition")],
        >>> ]
        >>> convertor = LevelConvertorWhitespace()
        >>> tokens2, entities2 = convertor.forward(text, entities)
        >>> chars3, entities3 = convertor.backward(tokens2, entities2)
    """
    def _convert(self, text):
        segments = re.split(r"( )", text)
        words = []; offset_mapping = []
        word_idx = 0; start = 0
        for seg in segments:
            if seg == " ":
                start += 1
                continue
            words.append(seg)
            word_idx += 1
            offset_mapping.append((start, start + len(seg)))
            start += len(seg)
        return words, offset_mapping
    
    def _forward(self, words):
        return " ".join(words)


class LevelConvertorHuggingFace(LevelConvertorBase):
    """
    Example:
        >>> text = "Unified Named Entity Recognition as Word-Word Relation Classification"
        >>> entities = [
        >>>     [(8, 8 + len("Named Entity Recognition"), "x", "Named Entity Recognition")],
        >>> ]
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        >>> convertor = LevelConvertorHuggingFace(tokenizer)
        >>> tokens2, entities2 = convertor.forward(text, entities)
        >>> chars3, entities3 = convertor.backward(tokens2, entities2)
    """

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def _convert(self, text):
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_tensors="np")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        offset_mapping = inputs["offset_mapping"][0].tolist()[1:-1]     # [CLS], [SEP]
        return tokens, offset_mapping
    
    def _forward(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens).strip()


class LevelConvertorHuggingFaceZh(LevelConvertorHuggingFace):

    def _convert(self, text):
        tokens = [ch if ch in self.tokenizer.vocab else 
            self.tokenizer.unk_token for ch in text]
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="np",)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        offset_mapping = [[i, i + 1] for i, ch in enumerate(tokens)]
        return tokens, offset_mapping


class ProcessConvertLevel(ProcessBase):

    def __init__(self, tokenizer, conversion, lang="en"):
        self.tokenizer = tokenizer
        self.conversion = conversion

        self.whitespace_converter = LevelConvertorWhitespace()
        if lang == "en":
            self.huggingface_converter = LevelConvertorHuggingFace(tokenizer)
        elif lang == "zh":
            self.huggingface_converter = LevelConvertorHuggingFaceZh(tokenizer)

        self.conversion2func_map = {
            "word2char": self.word_to_char,
            "char2word": self.char_to_word,
            "char2token": self.char_to_token,
            "token2char": self.token_to_char,
            "word2token": self.word_to_token,
            "token2word": self.token_to_word,
        }
        self.convert_func = self.conversion2func_map[self.conversion]
    
    def __call__(self, example):
        example = deepcopy(example)
        converted, entities = self.convert_func(
            example["text"], example["entities"])
        if "sent_start" in example and "sent_end" in example:
            sent_start, sent_end = example["sent_start"], example["sent_end"]
            sentence = [[(sent_start, sent_end, "_", example["text"][sent_start: sent_end])]]
            _, sentence = self.convert_func(example["text"], sentence)
            example["sent_start"], example["sent_end"] = sentence[0][0][:2]
        example["text"] = converted
        example["entities"] = entities
        return example
    
    def char_to_word(self, chars, char_level_entities):
        return self.whitespace_converter.forward(chars, char_level_entities)

    def word_to_char(self, words, word_level_entities):
        return self.whitespace_converter.backward(words, word_level_entities)
        
    def char_to_token(self, chars, char_level_entities):
        return self.huggingface_converter.forward(chars, char_level_entities)
    
    def token_to_char(self, tokens, token_level_entities):
        return self.huggingface_converter.backward(tokens, token_level_entities)
    
    def word_to_token(self, words, word_level_entities):
        return self.char_to_token(*self.word_to_char(words, word_level_entities))
    
    def token_to_word(self, tokens, token_level_entities):
        return self.char_to_word(*self.token_to_char(tokens, token_level_entities))

class ProcessMergeDiscontinuousSpans(ProcessBase):

    def __call__(self, example):
        if example["entities"] is None:
            return example

        example = deepcopy(example)
        entities = []
        for entity in example["entities"]:
            if len(entity) == 1:
                entities.append(entity)
            else:
                entities.append([])
                entity = sorted(entity, key=lambda x: x[:2])
                for span in entity:
                    start, end, label, string = span
                    if len(entities[-1]) == 0: 
                        entities[-1].append(span)
                        continue
                    last_start, last_end, last_label, last_string = entities[-1][-1]
                    if label != last_label or example["text"][last_end: start].strip() != "":
                        entities[-1].append(span)
                        continue
                    last_entity = entities[-1].pop(-1)
                    new_string = example["text"][last_start: end]
                    try:
                        assert new_string == last_string + " " + string
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    merged_span = (last_start, end, label, new_string)
                    entities[-1].append(merged_span)
                    # print(f"\nMerge {last_entity}, {span} to {merged_span}")
        example["entities"] = entities
        return example


class ProcessExample2Feature(ProcessBase):

    def __init__(self, label2id, tokenizer, max_sequence_length, 
            max_span_length, negative_sampling, keep_full_spans, stanza_nlp=None):
        super().__init__()
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_span_length = max_span_length
        self.negative_sampling = negative_sampling
        self.converter = ProcessConvertLevel(tokenizer, "char2token")
        self.keep_full_spans = keep_full_spans
        self.stanza_nlp = stanza_nlp
        if stanza_nlp is not None:
            self.stanza_upos_unit2id = self.stanza_nlp.processors['pos'].vocab._vocabs['upos']._unit2id
            
    def _encode_text(self, text: str):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_sequence_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

    def _encode_spans(self, input_length, sent_start, sent_end, word_ids):
        spans = []; spans_mask = []
        sent_start = min(sent_start, self.max_sequence_length - 1)
        sent_end = min(sent_end, self.max_sequence_length - 1)
        
        span_starts = np.arange(sent_start, sent_end) + 1       # (sequence_length,)
        span_lengths = np.arange(self.max_span_length)          # (input_length,)
        span_ends = span_starts.reshape(-1, 1) + span_lengths   # (sequence_length, input_length)
        span_starts = np.expand_dims(span_starts, 1) \
            .repeat(self.max_span_length, axis=1)               # (sequence_length, input_length)
        span_starts, span_ends = span_starts.reshape(-1), span_ends.reshape(-1)
        spans = np.stack([span_starts, span_ends], axis=-1)     # (sequence_length * input_length, 2)
        spans = spans[span_ends <= sent_end]                    # (num_spans, 2)
        
        if self.keep_full_spans:
            word_ids = word_ids[sent_start: sent_end]
            is_start = np.array(word_ids) != np.array([-1] + word_ids[:-1])
            is_end   = np.array(word_ids) != np.array(word_ids[1:] + [-1] )
            spans = spans[is_start[spans[:, 0] - 1] & is_end[spans[:, 1] - 1]]
        
        spans = [tuple(span) for span in spans.tolist()]
        spans_mask = np.ones(len(spans), dtype=np.int).tolist()
        return spans, spans_mask
    
    def _encode_syntactic(self, text, bert_offset_mapping):
        doc = self.stanza_nlp([text.split()]).to_dict()
        offset = 0
        upos = []; deprel = []
        stanza_offset_mapping = []
        for sent in doc:
            for token in sent:
                upos.append(token["upos"])
                deprel.append((
                    token["id"] + offset - 1, 
                    token["head"] - 1,
                    token["deprel"],
                ))
                stanza_offset_mapping.append((
                    token["start_char"],
                    token["end_char"],
                ))
            offset += len(sent)

        j = 0
        syntactic_upos_ids = []
        for i, (start, end) in enumerate(bert_offset_mapping):
            if start == end:
                syntactic_upos_ids.append(self.stanza_upos_unit2id["<PAD>"])
                continue
            syntactic_upos_ids.append(self.stanza_upos_unit2id[upos[j]])
            if end >= stanza_offset_mapping[j][1]:
                j += 1
        return syntactic_upos_ids

    def _encode_labels(self, entities, spans, input_length, offset_mapping):
        span2label_map = defaultdict(list)
        for entity in entities:
            # keep entities which are not truncated
            if any([start >= input_length or end >= start >= input_length for start, end, *_ in entity]):
                continue
            # span-to-label map(token-level)
            num_spans = len(entity)
            entity = sorted(entity, key=lambda x: (x[0], x[1]))
            for i, (start, end, label, string) in enumerate(entity):
                start, end = start + 1, end + 1                             # CLS, SEP
                start_prefix = "B" if i == 0 else "I"
                end_prefix = "E" if i == num_spans - 1 else "I"
                prefix = start_prefix + end_prefix
                span = (start, end - 1)
                label = prefix + "-" + label
                span2label_map[span].append(label)
            for span_a, span_b in zip(entity[:-1], entity[1:]):
                start, end = span_a[1], span_b[0]
                start, end = start + 1, end + 1                             # CLS, SEP
                span = (start, end - 1)
                label = "II" + "-" + "N"
                span2label_map[span].append(label)

        labels_onehot = []
        for span in spans:
            label_onehot = [self.label2id["O"]] * len(self.label2id)
            for label in span2label_map.get(span, []):
                label = self.label2id[label]
                label_onehot[label] = 1
            labels_onehot.append(label_onehot)
        return labels_onehot

    def __call__(self, example):
        text: str = example["text"]

        inputs = self._encode_text(text)
        word_ids = inputs.word_ids()[1:-1]
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        input_length = inputs["attention_mask"].sum().item()
        offset_mapping = inputs["offset_mapping"].numpy().tolist()

        example = self.converter(example)   # char -> token
        tokens, entities = example["text"], example["entities"]
        sent_start = example.get("sent_start", 0)
        sent_end = example.get("sent_end", len(tokens))

        # encode spans
        spans, spans_mask = self._encode_spans(input_length, sent_start, sent_end, word_ids)
        inputs["spans"], inputs["spans_mask"] = torch.tensor(spans), torch.tensor(spans_mask)

        # encode pos & depparse
        if self.stanza_nlp is not None:
            syntactic_upos_ids = self._encode_syntactic(text, offset_mapping)
            inputs["syntactic_upos_ids"] = torch.tensor(syntactic_upos_ids)

        if entities is None:
            inputs["labels"] = None
            return inputs

        labels = self._encode_labels(entities, spans, input_length - 2, offset_mapping)
        inputs["labels"] = torch.tensor(labels)  # (num_spans,)

        return inputs

class CadecProcessExample2Feature(ProcessExample2Feature):
    pass

class ShAReProcessExample2Feature(ProcessExample2Feature):
    pass


def load_dataset(data_class, process_class, data_name, data_dir, data_type, tokenizer, max_sequence_length, 
                 context_size, max_span_length, negative_sampling, keep_full_spans, stanza_nlp=None, **kwargs):
    process_piplines = [
        ProcessConvertLevel(tokenizer, "word2char") if data_class in [  # english
            CadecDataset, ShAReDataset,
        ] else None,
        ProcessMergeDiscontinuousSpans(),
        process_class(
            data_class.label2id(), tokenizer, max_sequence_length,
            max_span_length, negative_sampling, keep_full_spans, stanza_nlp,
        ),
    ]
    return data_class(data_name, data_dir, data_type, process_piplines, 
        context_size=context_size, max_examples=None, use_cache=True, **kwargs)

class XBilinear(nn.Module):

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, div: int = 1) -> None:
        super().__init__()
        self.features1_projection = nn.Linear(in1_features, in1_features // div)
        self.features2_projection = nn.Linear(in2_features, in2_features // div)
        self.bilinear = nn.Bilinear(in1_features // div, in2_features // div, out_features, bias=bias)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = self.features1_projection(input1)
        input2 = self.features2_projection(input2)
        return self.bilinear(input1, input2)

class XBiaffineRel(nn.Module):

    def __init__(self, hidden_size: int, out_features: int, bias: bool = True, div: int = 1) -> None:
        super().__init__()

        self.W1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // div, out_features))
        self.W2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // div))
        self.linear = nn.Linear(hidden_size + hidden_size + hidden_size // div, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
        relative_positions_embeddings = self._generate_relative_positions_embeddings(512, hidden_size // div)
        self.register_buffer("relative_positions_embeddings", relative_positions_embeddings)

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.W2.size(1))
        init.uniform_(self.W1, -bound, bound)
        init.uniform_(self.W2, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
    
    @classmethod
    def _generate_relative_positions_embeddings(cls, length, depth, max_relative_position=127):
        vocab_size = max_relative_position * 2 + 1
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        embeddings_table = np.zeros([vocab_size, depth])
        for pos in range(vocab_size):
            for i in range(depth // 2):
                embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
                embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

        embeddings_table_tensor = torch.tensor(embeddings_table).float()
        flat_relative_positions_matrix = final_mat.view(-1)
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(
            flat_relative_positions_matrix,num_classes=vocab_size).float()
        embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
        my_shape = list(final_mat.size())
        my_shape.append(depth)
        embeddings = embeddings.view(my_shape)
        return embeddings

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            input1: torch.Tensor[batch_size, num_spans, hidden_size]
            input2: torch.Tensor[batch_size, num_spans, hidden_size]
            spans: torch.Tensor[batch_size, num_spans, 2]
        
        """
        pe = self.relative_positions_embeddings[spans[..., 0], spans[..., 1]]
        # [x; y; p] W3
        output = self.linear(torch.cat([input1, input2, pe], dim=-1))
        # (x W1)(y W2 + p) / d
        input1 = torch.einsum("bnh,hdc->bndc", input1, self.W1)
        input2 = torch.einsum("bnh,hd->bnd", input2, self.W2)
        scale = math.sqrt(self.relative_positions_embeddings.size(-1))
        output = output + torch.einsum("bndc,bnd->bnc", input1, input2 + pe) / scale
        if self.bias is not None: 
            output = output + self.bias
        return output

class GCNConv(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        # x: (batch_size, num_nodes, in_features)
        # A: (batch_size, num_nodes, num_nodes)
        return F.relu(torch.einsum("bmn,bnh->bmh", A, self.linear(x)))

class GCN(nn.Module):

    def __init__(self, in_features, out_features, hidden_size, 
            num_layers=2, p=0.1, residual=False, symmetric=True):
        super().__init__()

        assert num_layers > 0
        self.layers = nn.ModuleList([])
        n_dims = in_features
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(n_dims, hidden_size))
            n_dims = hidden_size
        self.layers.append(GCNConv(n_dims, out_features))

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        self.symmetric = symmetric
        self.residual = residual

    def normalize(self, A, symmetric):
        # A = A+I
        A = A + torch.eye(A.size(0), device=A.device)
        # 所有节点的度
        d = A.sum(1)
        if symmetric:
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            D = D.mm(A).mm(D)
        else:
            # D=D^-1
            D = torch.diag(torch.pow(d, -1))
            D = D.mm(A)
        return D

    def forward(self, x, A):
        if self.residual:
            res = x
        A = torch.stack([
            self.normalize(a, self.symmetric) for a in A
        ], dim=0)
        for layer in self.layers:
            x = layer(x, A)
        x = self.dropout(x)
        if self.residual:
            x = x + res
        return x

class SpanClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_size,
                 do_projection=False, do_biaffine=False, do_biaffine_rel=False,
                 extract_method="endpoint", do_gcn=False, gcn_hidden_size=64, gcn_num_layers=2, 
                 gcn_dropout=0.1, gcn_residual=False):
        super().__init__()

        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_size)

        self.do_projection = do_projection
        if self.do_projection:
            self.start_projection = nn.Linear(hidden_size, hidden_size)
            self.end_projection   = nn.Linear(hidden_size, hidden_size)

        extract_method_func_map = {
            "endpoint": (
                self.forward_endpoint, 
                hidden_size * 2,
            ),
            "maxpool": (
                lambda sequence_output, spans: self.forward_pool(
                    sequence_output, spans, pool_method="max"), 
                hidden_size,
            ),
            "meanpool": (
                lambda sequence_output, spans: self.forward_pool(
                    sequence_output, spans, pool_method="mean"),
                hidden_size,
            ),
            "endpoint-pool": (
                self.forward_endpoint_pool,
                hidden_size * 4,
            ),
            "cln": (
                self.forward_cln,
                hidden_size * 2,
            )
        }
        self.forward, num_features = extract_method_func_map[extract_method]
        if extract_method == "cln":
            self.head2tail_cln = ConditionalLayerNorm(hidden_size, hidden_size)
            self.tail2head_cln = ConditionalLayerNorm(hidden_size, hidden_size)

        num_features += width_embedding_size

        self.do_gcn = do_gcn
        if self.do_gcn:
            self.gcn = GCN(num_features, num_features, gcn_hidden_size, 
                num_layers=gcn_num_layers, p=gcn_dropout, residual=gcn_residual, symmetric=True)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

        self.do_biaffine = do_biaffine
        if self.do_biaffine:
            self.bilinear = XBilinear(hidden_size, hidden_size, num_labels, bias=False, div=4)
        
        self.do_biaffine_rel = do_biaffine_rel
        if self.do_biaffine_rel:
            self.bilinear = XBiaffineRel(hidden_size, num_labels, bias=True, div=4)

    def _extract_spans_embedding_endpoint(self, sequence_output, spans):

        if self.do_projection:
            sequence_start_output = self.start_projection(sequence_output)
            sequence_end_output   = self.end_projection  (sequence_output)
        else:
            sequence_start_output = sequence_output
            sequence_end_output   = sequence_output

        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)

        spans_start_embedding = self.batched_index_select(sequence_start_output, spans_start)
        spans_end_embedding   = self.batched_index_select(sequence_end_output,   spans_end  )

        return spans_start_embedding, spans_end_embedding
    
    def _extract_spans_embedding_pool(self, sequence_output, spans, pool_method):

        batch_size, sequence_length, hidden_size = sequence_output.size()
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)
        spans_width = spans_end - spans_start + 1

        # span indices & mask
        max_span_width = spans_width.max().item()
        max_span_range_indices = torch.arange(0, max_span_width, 
            dtype=torch.long, device=sequence_output.device).view(1, 1, -1)     # (1, 1, max_span_width)
        raw_span_indices = spans_start.unsqueeze(-1) + max_span_range_indices   # (batch_size, num_spans, max_span_width)

        span_mask = max_span_range_indices <= spans_width.unsqueeze(-1)         # (batch_size, num_spans, max_span_width)
        span_mask = span_mask & (raw_span_indices < sequence_length) & (0 <= raw_span_indices)
        span_indices = raw_span_indices * span_mask                             # (batch_size, num_spans, max_span_width)

        # batched index select
        span_indices_flatten = span_indices.view(batch_size, -1)                # (batch_size, num_spans * max_span_width)
        span_indices_embedding = self.batched_index_select(
            sequence_output, span_indices_flatten)                              # (batch_size, num_spans * max_span_width, hidden_size)
        span_indices_embedding = span_indices_embedding.view(
            batch_size, -1, max_span_width, hidden_size)                        # (batch_size, num_spans, max_span_width, hidden_size)
        
        # masked max/mean
        extend_span_mask = span_mask.unsqueeze(-1)                              # (batch_size, num_spans, max_span_width, 1)
        if pool_method == "max":
            span_indices_embedding.masked_fill_(
                ~extend_span_mask, torch.finfo(sequence_output.dtype).min)                         
            span_indices_embedding, _ = span_indices_embedding.max(dim=2)       # (batch_size, num_spans, hidden_size)
        elif pool_method == "mean":
            span_indices_embedding = span_indices_embedding * extend_span_mask
            span_indices_embedding = span_indices_embedding.sum(dim=2) / extend_span_mask.sum(dim=2)
        
        return span_indices_embedding
    
    def _extract_width_embedding(self, spans):
        
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end   = spans[:, :, 1].view(spans.size(0), -1)
        spans_width = spans_end - spans_start + 1

        return self.width_embedding(spans_width)

    def forward_endpoint(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            spans_start_embedding, 
            spans_end_embedding,
            spans_width_embedding,
        ], dim=-1)  # (batch_size, num_spans, num_features)

        if self.do_gcn:
            batch_size, num_spans, _ = spans_embedding.size()

            # 仅相邻的片段间：(s1, e1)(s2, e2), (s2, e2)(s1, e1)
            spans_mask = ((spans[..., 0] > 0) & (spans[..., 1] > 0)).long()
            starts = spans[..., 0]; ends = spans[..., 1]    # (batch_size, num_spans)
            adjacent_matrix = (spans_mask.unsqueeze(-1) * spans_mask.unsqueeze(-2)) & \
                ((ends.unsqueeze(-1) + 1 == starts.unsqueeze(-2)) | (starts.unsqueeze(-1) == ends.unsqueeze(-2) + 1))
            adjacent_matrix = adjacent_matrix.float()

            # gcn
            spans_embedding = spans_embedding + self.gcn(spans_embedding, adjacent_matrix)

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding)
        if self.do_biaffine_rel:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding, spans)

        return logits

    def forward_pool(self, sequence_output, spans, pool_method):
        
        spans_width_embedding = self._extract_width_embedding(spans)

        spans_pool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, pool_method),
        spans_embedding = torch.cat([
            spans_pool_embedding,
            spans_width_embedding,
        ], dim=-1)             # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)

        return logits

    def forward_endpoint_pool(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_meanpool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, "mean")
        spans_maxpool_embedding = self._extract_spans_embedding_pool(
            sequence_output, spans, "max")
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            spans_start_embedding,
            spans_end_embedding,
            spans_meanpool_embedding,
            spans_maxpool_embedding,
            spans_width_embedding
        ], dim=-1)  # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding)

        return logits

    def forward_cln(self, sequence_output, spans):

        spans_start_embedding, spans_end_embedding = \
            self._extract_spans_embedding_endpoint(sequence_output, spans)
        spans_width_embedding = self._extract_width_embedding(spans)
        
        spans_embedding = torch.cat([
            self.tail2head_cln(spans_start_embedding.unsqueeze(1), spans_end_embedding).squeeze(1),
            self.head2tail_cln(spans_end_embedding.unsqueeze(1), spans_start_embedding).squeeze(1),
            spans_width_embedding
        ], dim=-1)  # (batch_size, num_spans, num_features)

        logits = self.classifier(spans_embedding)
        if self.do_biaffine:
            logits = logits + self.bilinear(spans_start_embedding, spans_end_embedding)

        return logits

    @classmethod
    def batched_index_select(cls, input, index):
        batch_size, sequence_length, hidden_size = input.size()
        index_onehot = F.one_hot(index, num_classes=sequence_length).float()
        output = torch.bmm(index_onehot, input)
        return output

    @classmethod
    def decode(cls, logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits=True, do_post_processing=False):
        other_id = label2id["O"]
        if not isinstance(thresh, (List, Tuple)):
            thresh = [thresh]
        if len(thresh) == 1:
            thresh *= len(label2id)
        assert len(thresh) == len(label2id)
        if is_logits:
            labels = logits_or_labels.sigmoid()
        else:
            labels = logits_or_labels

        batch_size = logits_or_labels.size(0)
        spans = spans.cpu().numpy().tolist()
        spans_mask = spans_mask.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()

        def add_IIN_spans(raw_spans):
            """
            这一步的目的是处理模型输出，多个片段的连续实体的情况，如
            [
                ((13, "B"), (15, "I"), "ADR"),
                ((15, "I"), (18, "E"), "ADR")
            ]
            缺少IIN片段((15, "I"), (15, "I"), "N")，导致后续步骤解码失败
            """
            iin_spans = []
            for i in range(len(raw_spans)):
                if raw_spans[i][-1] == "N":
                    continue
                for j in range(i + 1, len(raw_spans)):
                    if raw_spans[i][-1] != raw_spans[j][-1]:
                        continue
                    if raw_spans[i][1] == raw_spans[j][0]:
                        iin_spans.append((raw_spans[i][1], raw_spans[i][1], "N"))
                    if raw_spans[j][1] == raw_spans[i][0]:
                        iin_spans.append((raw_spans[j][1], raw_spans[j][1], "N"))
            raw_spans.extend(iin_spans)
            return raw_spans
            
        def decode_support(raw_spans, current_entities, index=None, start=None, prefix=None, label=None):
            if current_entities and current_entities[index][-1][1][1] == "E":
                return current_entities
            current_entity = current_entities.pop(index) if index is not None else None
            for span_no, span in enumerate(raw_spans):
                (start_pos, start_prefix), (end_pos, end_prefix), span_label = span
                if start is None and start_prefix == "B" and label != "N":
                    current_entities.append([span])                     # 新建实体
                    next_start, next_prefix, next_label = end_pos, end_prefix, "N"
                    current_entities = decode_support(
                        raw_spans, current_entities, len(current_entities) - 1,
                        next_start, next_prefix, next_label,
                    )
                else:
                    if start_pos == start and start_prefix == prefix and span_label == label:
                        new_entity = deepcopy(current_entity) + [span]  # 在index对应实体上追加
                        current_entities.append(new_entity)
                        next_start, next_prefix = end_pos, end_prefix
                        next_label = "N" if label != "N" else new_entity[0][-1] # label/N 交替出现
                        current_entities = decode_support(
                            raw_spans, current_entities, len(current_entities) - 1,
                            next_start, next_prefix, next_label,
                        )
            return current_entities

        decode_spans = []; decode_entities = []
        for i in range(batch_size):

            raw_spans = []  # 第i样本的span
            sent_start = float("inf")
            for span, mask, label in zip(spans[i], spans_mask[i], labels[i]):
                if mask == 0: break
                start, end = span  # 左闭右闭
                sent_start = min(sent_start, start)
                start -= sent_start; end -= sent_start
                for label_id, label_proba in enumerate(label):
                    if label_proba < thresh[label_id]:
                        continue
                    if id2label[label_id] == "O": continue
                    (start_prefix, end_prefix), label = id2label[label_id].split("-")
                    raw_spans.append(((start, start_prefix), (end + 1, end_prefix), label))  # 左闭右开
            raw_spans = sorted(raw_spans, key=lambda x: (x[0][0], x[1][0]))  # 以实体出现位置排序
            decode_spans.append(raw_spans)

            # 组合
            raw_spans = add_IIN_spans(raw_spans)
            raw_entities = decode_support(raw_spans, [])

            # 实体
            entities = []
            for raw_entity in raw_entities:
                entities.append([])
                for raw_span in raw_entity:
                    (start_pos, start_prefix), (end_pos, end_prefix), label = raw_span
                    if label == "N": continue
                    entities[-1].append((start_pos, end_pos, label))
            
            # 去重
            entities = set([tuple(entity) for entity in entities])
            entities = [[span for span in entity] for entity in entities]
            
            # 排序
            if len(entities) > 0:
                entities = sorted(entities, key=lambda x: (x[0][0], x[0][1] - x[0][0]))
            decode_entities.append(entities)

        return decode_spans, decode_entities


class MultilabelCategoricalCrossEntropy(nn.Module):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true, mask=None):
        # y_pred: tensor(batch_size * num_labels, num_samples)
        # y_true: tensor(batch_size * num_labels, num_samples)
        # mask[Optional]: tensor(batch_size * num_labels, num_samples)
        if mask is None: mask = torch.ones_like(y_true, dtype=torch.long)
        valid_mask = (y_true != self.ignore_index) & (mask > 0)

        # mask
        y_pred = torch.where(y_true == 0, y_pred, - y_pred)
        ninf = torch.full_like(y_pred, -1e12)
        y_pred_neg = torch.where((y_true == 0) & valid_mask, y_pred, ninf)
        y_pred_pos = torch.where((y_true == 1) & valid_mask, y_pred, ninf)

        # 补充全1列
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        # 计算损失
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()


class SpanClassificationLoss(nn.Module):

    def __init__(
        self, 
        num_labels, 
        loss_type="ce",
        label_smoothing_eps=0.0, 
        focal_gamma=2.0, 
        focal_alpha=0.25,
        reduction="mean", 
        ignore_index=-100
    ):
        super().__init__()
        self.reduction = reduction
        if loss_type == "ce":
            self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        elif loss_type == "multicate":
            self.loss_fct = MultilabelCategoricalCrossEntropy(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        logits,  # (batch_size, num_spans, num_labels)
        labels,  # (batch_size, num_spans, num_labels)
        mask,    # (batch_size, num_spans,)
    ):
        # batch_size = logits.size(0)
        if isinstance(self.loss_fct, MultilabelCategoricalCrossEntropy):
            batch_size, num_spans, num_labels = logits.size()
            mask = mask.unsqueeze(-1).expand_as(labels)
            logits = logits.transpose(2, 1).contiguous().view(-1, num_spans)
            labels = labels.transpose(2, 1).contiguous().view(-1, num_spans)
            mask = mask.transpose(2, 1).contiguous().view(-1, num_spans)
            return self.loss_fct(logits, labels.float(), mask)
        loss = self.loss_fct(logits, labels.float())
        activate_mask = (mask > 0).unsqueeze(-1).expand_as(loss)
        loss = loss[activate_mask]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class SpanClassificationOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    groundtruths: List[List[Entity]] = None
    predictions: List[List[Entity]] = None
    predict_spans: List[List[Span]] = None
    groundtruth_spans: List[List[Span]] = None


class ModelForSpanClassification(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(
            config, "classifier_dropout") else config.hidden_dropout_prob)
        
        if config.conditional:
            self.conditional_embeddings = nn.Embedding(2, self.config.hidden_size)
            self.conditional_layer_norm = ConditionalLayerNorm(
                self.config.hidden_size, self.config.hidden_size)
        
        if config.do_lstm:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, 
                num_layers=config.num_lstm_layers, batch_first=True, bidirectional=True)

        self.head = SpanClassificationHead(
            config.hidden_size, config.num_labels,
            config.max_span_length, config.width_embedding_size,
            config.do_projection, 
            config.do_biaffine, config.do_biaffine_rel,
            config.extract_method,
            config.do_gcn, config.gcn_hidden_size,
            config.gcn_num_layers, config.gcn_dropout,
            config.gcn_residual,
        )

        self.loss_fct = SpanClassificationLoss(
            num_labels=config.num_labels, 
            loss_type=config.loss_type,
            label_smoothing_eps=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        syntactic_upos_ids=None,
        conditional_ids=None,
        spans=None,
        spans_mask=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = self.config.use_last_n_layers is not None

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_last_n_layers is not None:
            last_hidden_state = outputs["hidden_states"]
            last_hidden_state = torch.stack(last_hidden_state[- self.config.use_last_n_layers: ], dim=-1)
            if self.config.agg_last_n_layers == "mean":
                last_hidden_state = torch.mean(last_hidden_state, dim=-1)
            elif self.config.agg_last_n_layers == "sum":
                last_hidden_state = torch.sum(last_hidden_state, dim=-1)
            elif self.config.agg_last_n_layers == "max":
                last_hidden_state = torch.max(last_hidden_state, dim=-1)[0]
            outputs["last_hidden_state"] = last_hidden_state

        sequence_output = outputs["last_hidden_state"]

        if self.config.do_lstm:
            sequence_lengths = attention_mask.sum(dim=1)
            packed_sequence_output = pack_padded_sequence(
                sequence_output, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_sequence_output, _ = self.lstm(packed_sequence_output)
            unpacked_sequence_output, _ = pad_packed_sequence(
                packed_sequence_output, batch_first=True, total_length=sequence_lengths.max())
            sequence_output = sequence_output + unpacked_sequence_output

        sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output, spans)          # (batch_size, num_spans, num_labels)
        predict_spans = predictions = None
        if not self.training:
            predict_spans, predictions = self.decode(logits, spans, spans_mask, self.config.decode_thresh,
                                      self.config.label2id, self.config.id2label, is_logits=True,
                                      do_post_processing=self.config.do_post_processing)

        loss = None
        groundtruth_spans = groundtruths = None
        if labels is not None:
            loss = self.loss_fct(logits, labels, spans_mask)
            if not self.training:
                groundtruth_spans, groundtruths = self.decode(labels, spans, spans_mask, self.config.decode_thresh,
                                        self.config.label2id, self.config.id2label, is_logits=False,
                                        do_post_processing=self.config.do_post_processing)

        if not return_dict:
            outputs = (logits, predictions, groundtruths) + outputs[2:]
            return ((loss,) + outputs) if loss is not None else outputs

        return SpanClassificationOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            groundtruths=groundtruths,
            predict_spans=predict_spans,
            groundtruth_spans=groundtruth_spans,
        )

    @classmethod
    def decode(cls, logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits=True, do_post_processing=False):
        return SpanClassificationHead.decode(logits_or_labels, spans, spans_mask, thresh, label2id, id2label, is_logits, do_post_processing)


class BertForSpanClassification(BertPreTrainedModel, ModelForSpanClassification):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()


class RobertaForSpanClassification(RobertaPreTrainedModel, ModelForSpanClassification):

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.init_weights()


def precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    *,
                                    average: Optional[str] = None,
                                    labels: Optional[List[str]] = None,
                                    entity_type: str = "all",
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False):
    """Compute precision, recall, F-measure and support for each class.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a tagger.

        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.

        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "warn", this acts as 0, but warnings are also raised.

        suffix : bool, False by default.

    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]

        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]

        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]

        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.

    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):

            if entity_type == "continuous":
                true = [entity for entity in true if len(entity) == 1]
                pred = [entity for entity in pred if len(entity) == 1]
            if entity_type == "discontinuous":
                true = [entity for entity in true if len(entity) > 1]
                pred = [entity for entity in pred if len(entity) > 1]
            is_contain_discontinuous = any([len(spans) > 1 for spans in true])
            if entity_type == "examples_without_discontiguous" and is_contain_discontinuous:
                continue    # 仅保留只有连续实体的样本
            if entity_type == "examples_with_discontinuous" and not is_contain_discontinuous:
                continue    # 仅保留包含不连续实体的样本
            if entity_type == "without_label":
                true = [[(start, end, "_") for start, end, _ in entity] for entity in true]
                pred = [[(start, end, "_") for start, end, _ in entity] for entity in pred]

            for spans in true:
                type_name = spans[0][2]
                assert all([type_name == label for _, _, label in spans])
                tokens = set()
                for start, end, label in spans:
                    for j in range(start, end):
                        tokens.add(j)
                entities_true[type_name].add((i, frozenset(tokens)))

            for spans in pred:
                type_name = spans[0][2]
                assert all([type_name == label for _, _, label in spans])
                tokens = set()
                for start, end, label in spans:
                    for j in range(start, end):
                        tokens.add(j)
                entities_pred[type_name].add((i, frozenset(tokens)))

        if labels is not None:
            entities_true = {k: v for k, v in entities_true.items() if k in labels}
            entities_pred = {k: v for k, v in entities_pred.items() if k in labels}

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        # print(f"#tp_sum={tp_sum}, #pred_sum={pred_sum}, #true_sum={true_sum}")
        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


class SequenceLabelingScoreEntity(SequenceLabelingScore):

    def __init__(self, labels, average="micro", entity_type="all"):
        super().__init__(labels, average)
        self.entity_type = entity_type

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []

        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average,
                labels=None if label == self.average else [label],
                entity_type=self.entity_type
            )
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            f"df_{self.name()}": df, f"f1_{self.average}_{self.name()}": f1,   # for monitor
        }

    def name(self):
        return f"{self.entity_type}_entity"


class SequenceLabelingScoreSpan(SequenceLabelingScore):

    def __init__(self, labels, average="micro", span_length=None):
        super().__init__(labels, average)
        self.span_length = span_length

    def update(self, 
        preds: List[List[Span]],
        target: List[List[Span]],
    ):
        for pred_spans, target_spans in zip(preds, target):
            self.preds.append([])
            self.target.append([])
            for (start, start_prefix), (end, end_prefix), label in pred_spans:
                span = (start, end, f"{start_prefix}{end_prefix}-{label}")
                span_length = end - start
                if self.span_length is None or span_length == self.span_length:
                    self.preds[-1].append(span)
            for (start, start_prefix), (end, end_prefix), label in target_spans:
                span = (start, end, f"{start_prefix}{end_prefix}-{label}")
                span_length = end - start
                if self.span_length is None or span_length == self.span_length:
                    self.target[-1].append(span)
                
    def value(self):
        res = super().value()
        return {
            f"df_{self.name()}": res["df"], 
            f"f1_{self.average}_{self.name()}": res[f"f1_{self.average}"],   # for monitor
        }

    def name(self):
        if self.span_length is None:
            return "span"
        return f"span-{self.span_length}"


class Trainer(TrainerBase):

    def evaluate(self, dev_data, prefix_metric=None, save_dir=None, save_result=False, file_name=None, multi_thresh=True):
        '''
        Evaluate the model on a validation set
        '''
        all_batch_list = []
        eval_dataloader = self.build_eval_dataloader(dev_data)
        self.build_record_tracker()
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if batch.get('loss'):
                self.records['loss_meter'].update(batch['loss'], n=1)

            # FIXED: groundtruth不应经encode/decode后得到，会有错误,但是cadec可以复现0.735
            groundtruths = batch["groundtruths"]
            start_index = eval_dataloader.batch_size * step
            end_index   = eval_dataloader.batch_size * (step + 1)
            examples = dev_data.examples[start_index: end_index]
            for i, example in enumerate(examples):
                ## raw examples -> token level examples
                for proc in dev_data.process_piplines[:-1]:
                    if proc is None: continue
                    example = proc(example)
                example = dev_data.process_piplines[-1].converter(example)
                groundtruths[i] = [
                    [(start, end, label) for start, end, label, string in entity] 
                    for entity in example["entities"]
                ]

            all_batch_list.append(batch)
            pbar.step(step)
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg

        # find best thresh
        if self.opts.find_best_decode_thresh and not multi_thresh:
            best_metric_value = - np.inf
            best_decode_thresh = None
            # decode_threshs = np.linspace(0.1, 0.9, 19).tolist()
            decode_threshs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            pbar = tqdm(decode_threshs, total=len(decode_threshs))
            for decode_thresh in pbar:
                self.reset_metrics()
                all_batch_list_temp = []
                for batch_list in all_batch_list:
                    all_batch_list_temp.append(dict())
                    logits = batch_list["logits"]
                    spans  = batch_list["spans" ]
                    spans_mask = batch_list["spans_mask"]
                    groundtruths = batch_list["groundtruths"]
                    _, predictions = self.model.decode(logits, spans, spans_mask, decode_thresh,
                        self.model.config.label2id, self.model.config.id2label, is_logits=True,
                        do_post_processing=self.model.config.do_post_processing)
                    all_batch_list_temp[-1]["predictions"] = predictions
                    all_batch_list_temp[-1]["groundtruths"] = groundtruths
                    all_batch_list_temp[-1]["predict_spans"] = batch_list["predict_spans"]
                    all_batch_list_temp[-1]["groundtruth_spans"] = batch_list["groundtruth_spans"]
                self.update_metrics(all_batch_list_temp, prefix_metric)
                for metric in self.metrics:
                    if not isinstance(metric, SequenceLabelingScoreEntity):
                        continue
                    metric_value = metric.value()
                    key = self.opts.checkpoint_monitor.split("_", 1)[1]
                    if key not in metric_value:
                        continue
                    metric_value = metric_value[key]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_decode_thresh = decode_thresh
                        for batch_list, batch_list_temp in zip(all_batch_list, all_batch_list_temp):
                            batch_list["predictions"] = batch_list_temp["predictions"]
                            batch_list["best_decode_thresh"] = best_decode_thresh
                pbar.set_description(f"[All] Finding best decode threshold ({best_decode_thresh:.2f}/{best_metric_value:.6f})...")
            self.logger.info(f"Best decde threshold is {best_decode_thresh:.6f}, best metric value is {best_metric_value}.")

        # find best thresh
        if self.opts.find_best_decode_thresh and multi_thresh:

            # 连续实体阈值搜索
            best_decode_thresh = None
            best_metric_value = - np.inf
            decode_threshs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            pbar = tqdm(decode_threshs, total=len(decode_threshs))
            for decode_thresh in pbar:
                decode_thresh_list = [
                    decode_thresh if label[:2] in ["BE", ] else 1.0
                    for label, id in self.opts.label2id.items()
                ]
                self.reset_metrics()
                all_batch_list_temp = []
                for batch_list in all_batch_list:
                    all_batch_list_temp.append(dict())
                    logits = batch_list["logits"]
                    spans  = batch_list["spans" ]
                    spans_mask = batch_list["spans_mask"]
                    groundtruths = batch_list["groundtruths"]
                    _, predictions = self.model.decode(logits, spans, spans_mask, decode_thresh_list,
                        self.model.config.label2id, self.model.config.id2label, is_logits=True,
                        do_post_processing=self.model.config.do_post_processing)
                    all_batch_list_temp[-1]["predictions"] = predictions
                    all_batch_list_temp[-1]["groundtruths"] = groundtruths
                    all_batch_list_temp[-1]["predict_spans"] = batch_list["predict_spans"]
                    all_batch_list_temp[-1]["groundtruth_spans"] = batch_list["groundtruth_spans"]
                self.update_metrics(all_batch_list_temp, prefix_metric)
                for metric in self.metrics:
                    if not isinstance(metric, SequenceLabelingScoreEntity):
                        continue
                    metric_value = metric.value()
                    key = "f1_micro_continuous_entity"
                    if key not in metric_value:
                        continue
                    metric_value = metric_value[key]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_decode_thresh = decode_thresh
                pbar.set_description(f"[Contiguous] Finding best decode threshold ({best_decode_thresh:.2f}/{best_metric_value:.6f})...")
            best_decode_thresh_cont = best_decode_thresh

            # 非连续实体阈值搜索
            best_decode_thresh = None
            best_metric_value = - np.inf
            decode_threshs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            pbar = tqdm(decode_threshs, total=len(decode_threshs))
            for decode_thresh in pbar:
                decode_thresh_list = [
                    decode_thresh if label[:2] in ["BI", "II", "IE"] else 1.0
                    for label, id in self.opts.label2id.items()
                ]
                self.reset_metrics()
                all_batch_list_temp = []
                for batch_list in all_batch_list:
                    all_batch_list_temp.append(dict())
                    logits = batch_list["logits"]
                    spans  = batch_list["spans" ]
                    spans_mask = batch_list["spans_mask"]
                    groundtruths = batch_list["groundtruths"]
                    _, predictions = self.model.decode(logits, spans, spans_mask, decode_thresh_list,
                        self.model.config.label2id, self.model.config.id2label, is_logits=True,
                        do_post_processing=self.model.config.do_post_processing)
                    all_batch_list_temp[-1]["predictions"] = predictions
                    all_batch_list_temp[-1]["groundtruths"] = groundtruths
                    all_batch_list_temp[-1]["predict_spans"] = batch_list["predict_spans"]
                    all_batch_list_temp[-1]["groundtruth_spans"] = batch_list["groundtruth_spans"]
                self.update_metrics(all_batch_list_temp, prefix_metric)
                for metric in self.metrics:
                    if not isinstance(metric, SequenceLabelingScoreEntity):
                        continue
                    metric_value = metric.value()
                    key = "f1_micro_discontinuous_entity"
                    if key not in metric_value:
                        continue
                    metric_value = metric_value[key]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_decode_thresh = decode_thresh
                pbar.set_description(f"[Discontiguous] Finding best decode threshold ({best_decode_thresh:.2f}/{best_metric_value:.6f})...")
            best_decode_thresh_disc = best_decode_thresh

            # 最优阈值下计算指标 & 更新结果
            self.logger.info(f"Best decode threshold is {best_decode_thresh_cont:.6f}(cont)/{best_decode_thresh_disc:.6f}(disc), best metric value is {best_metric_value}.")
            decode_thresh_list = [
                best_decode_thresh_cont if label.startswith("BE") else best_decode_thresh_disc
                for label, id in self.opts.label2id.items()
            ]
            self.reset_metrics()
            all_batch_list_temp = []
            for batch_list in all_batch_list:
                all_batch_list_temp.append(dict())
                logits = batch_list["logits"]
                spans = batch_list["spans"]
                spans_mask = batch_list["spans_mask"]
                groundtruths = batch_list["groundtruths"]
                _, predictions = self.model.decode(logits, spans, spans_mask, decode_thresh_list,
                    self.model.config.label2id, self.model.config.id2label, is_logits=True,
                    do_post_processing=self.model.config.do_post_processing)
                all_batch_list_temp[-1]["predictions"] = predictions
                all_batch_list_temp[-1]["groundtruths"] = groundtruths
                all_batch_list_temp[-1]["predict_spans"] = batch_list["predict_spans"]
                all_batch_list_temp[-1]["groundtruth_spans"] = batch_list["groundtruth_spans"]
            self.update_metrics(all_batch_list_temp, prefix_metric)
            for metric in self.metrics:
                if not isinstance(metric, SequenceLabelingScoreEntity):
                    continue
                metric_value = metric.value()
                key = "f1_micro_all_entity"
                if key not in metric_value:
                    continue
                metric_value = metric_value[key]
                best_decode_thresh = decode_thresh
                for batch_list, batch_list_temp in zip(all_batch_list, all_batch_list_temp):
                    batch_list["predictions"] = batch_list_temp["predictions"]
                    batch_list["best_decode_thresh"] = best_decode_thresh

        self.reset_metrics()
        self.update_metrics(all_batch_list, prefix_metric)
        self.print_evaluate_result()
        if save_result:
            if file_name is None: file_name = f"dev_eval_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_metrics(self, all_batch_list, prefix):
        preds = []; target = []
        preds_span = []; target_span = []
        for batch in all_batch_list:
            preds.extend(batch["predictions"])
            target.extend(batch["groundtruths"])
            preds_span.extend(batch["predict_spans"])
            target_span.extend(batch["groundtruth_spans"])
        prefix = "" if prefix is None else prefix + "_"
        for metric in self.metrics:
            if isinstance(metric, SequenceLabelingScoreSpan):
                metric.update(preds=preds_span, target=target_span)
            else:
                metric.update(preds=preds, target=target)
            value = metric.value()
            if value:
                if isinstance(value, float):
                    self.records["result"][f"{prefix}eval_{metric.name()}"] = value
                elif isinstance(value, dict):
                    self.records["result"].update({f"{prefix}eval_{k}": v for k, v in value.items()})
                else:
                    raise ValueError("metric value type: expected one of (float, dict)")
            else:
                self.logger.info(f"{metric.name()} value is None")

    def print_evaluate_result(self):
        '''
        打印evaluation结果
        '''
        if len(self.records['result']) == 0:
            self.logger.warning("eval result record is empty")
        self.logger.info("***** Evaluating results of %s *****", self.opts.task_name)
        self.logger.info("  global step = %s", self.global_step)
        metric_key_value_map = dict()
        for key in sorted(self.records['result'].keys()):
            if isinstance(self.records['result'][key], pd.DataFrame):
                self.logger.info("  %s = \n%s", key, str(round(self.records['result'][key], 4)))
            else:
                self.logger.info("  %s = %s", key, str(round(self.records['result'][key], 4)))
                name = key.split("_")[1] if "_" in key else key
                self.writer.add_scalar(f"{name[0].upper() + name[1:]}/{key}", self.records['result'][key],
                                       int(self.global_step / self.opts.logging_steps))
                metric_key_value_map[key] = self.records['result'][key]
        if use_wandb:
            wandb.log(metric_key_value_map)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSpanClassification, BertTokenizerFast),
    "roberta": (RobertaConfig, RobertaForSpanClassification, RobertaTokenizerFast),
}

DATA_CLASSES = {
    "cadec": (CadecDataset, CadecProcessExample2Feature),
    "share13": (ShAReDataset, ShAReProcessExample2Feature),
    "share14": (ShAReDataset, ShAReProcessExample2Feature),
}


def build_opts():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        opts = Argparser.parse_args_from_json(json_file=os.path.abspath(sys.argv[1]))
    else:
        parser = Argparser.get_training_parser()
        group = parser.add_argument_group(title="user-defined", description="user-defined")
        group.add_argument("--labels", nargs="+", type=str, default=None)
        group.add_argument("--context_size", type=int, default=0)
        group.add_argument("--classifier_dropout", type=float, default=0.1)
        group.add_argument("--use_last_n_layers", type=int, default=None)
        group.add_argument("--agg_last_n_layers", type=str, default="mean")
        group.add_argument("--negative_sampling", type=float, default=0.0)
        group.add_argument("--max_span_length", type=int, default=30)
        group.add_argument("--keep_full_spans", action="store_true")
        group.add_argument("--width_embedding_size", type=int, default=128)
        group.add_argument("--loss_type", type=str, default="lsr", choices=["ce", "lsr", "focal", "multicate"])
        group.add_argument("--label_smoothing", type=float, default=0.0)
        group.add_argument("--focal_gamma", type=float, default=2.0)
        group.add_argument("--focal_alpha", type=float, default=0.25)
        group.add_argument("--decode_thresh", type=float, nargs="+", default=0.5)
        group.add_argument("--extract_method", type=str, default="endpoint")
        group.add_argument("--do_gcn", action="store_true")
        group.add_argument("--gcn_hidden_size", type=int, default=64)
        group.add_argument("--gcn_num_layers", type=int, default=2)
        group.add_argument("--gcn_dropout", type=float, default=0.1)
        group.add_argument("--gcn_residual", action="store_true")
        group.add_argument("--find_best_decode_thresh", action="store_true")
        group.add_argument("--use_sinusoidal_width_embedding", action="store_true")
        group.add_argument("--do_lstm", action="store_true")
        group.add_argument("--num_lstm_layers", type=int, default=1)
        group.add_argument("--do_projection", action="store_true")
        group.add_argument("--do_biaffine", action="store_true")
        group.add_argument("--do_biaffine_rel", action="store_true")
        group.add_argument("--do_post_processing", action="store_true")
        opts = parser.parse_args_from_parser(parser)
    
    return opts

def main(opts):

    logger = Logger(opts=opts)
    if use_wandb:
        # Initialize wandb
        if not wandb.run:   # for hyperparameter search
            wandb.init(
                project="discontinuous-ner", 
                group=None,
                name=opts.experiment_code,
                tags=[opts.task_name, opts.model_type, "span"],
                config=opts, 
            )

    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    data_class, process_class = DATA_CLASSES[opts.task_name]

    # data processor
    logger.info("initializing data processor")
    stanza_nlp = None
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    train_dataset = load_dataset(data_class, process_class, opts.train_input_file, opts.data_dir, "train",
                                 tokenizer, opts.train_max_seq_length, opts.context_size, opts.max_span_length, 
                                 opts.negative_sampling, opts.keep_full_spans, stanza_nlp=stanza_nlp, labels=opts.labels)
    dev_dataset   = load_dataset(data_class, process_class, opts.eval_input_file, opts.data_dir, "dev",
                                 tokenizer, opts.eval_max_seq_length, opts.context_size, opts.max_span_length,
                                 opts.negative_sampling, opts.keep_full_spans, stanza_nlp=stanza_nlp, labels=opts.labels)
    test_dataset  = load_dataset(data_class, process_class, opts.test_input_file, opts.data_dir, "test",
                                 tokenizer, opts.test_max_seq_length, opts.context_size, opts.max_span_length,
                                 opts.negative_sampling, opts.keep_full_spans, stanza_nlp=stanza_nlp, labels=opts.labels)

    if stanza_nlp is not None and train_dataset.use_cache and dev_dataset.use_cache and test_dataset.use_cache:
        del stanza_nlp
    opts.num_labels = train_dataset.num_labels
    opts.label2id = data_class.label2id()
    opts.id2label = data_class.id2label()

    # model
    logger.info("initializing model and config")
    config, unused_kwargs = config_class.from_pretrained(
        opts.pretrained_model_path, return_unused_kwargs=True,
        num_labels=opts.num_labels, id2label=opts.id2label, label2id=opts.label2id,
        conditional=False,
        classifier_dropout=opts.classifier_dropout, 
        use_last_n_layers=opts.use_last_n_layers,
        agg_last_n_layers=opts.agg_last_n_layers,
        negative_sampling=opts.negative_sampling,
        max_span_length=opts.max_span_length, 
        width_embedding_size=opts.width_embedding_size,
        loss_type=opts.loss_type,
        label_smoothing=opts.label_smoothing, 
        focal_gamma=opts.focal_gamma,
        focal_alpha=opts.focal_alpha,
        decode_thresh=opts.decode_thresh,
        do_lstm=opts.do_lstm, 
        num_lstm_layers=opts.num_lstm_layers,
        do_projection=opts.do_projection,
        extract_method=opts.extract_method,
        do_gcn=opts.do_gcn,
        gcn_hidden_size=opts.gcn_hidden_size,
        gcn_num_layers=opts.gcn_num_layers, 
        gcn_dropout=opts.gcn_dropout,
        gcn_residual=opts.gcn_residual,
        do_biaffine=opts.do_biaffine,
        do_biaffine_rel=opts.do_biaffine_rel,
        do_post_processing=opts.do_post_processing,
    )
    for key, value in unused_kwargs.items():
        setattr(config, key, value)

    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)

    def update_example_entities(tokenizer, examples, entities, process):
        assert len(examples) == len(entities)
        updated = []
        for example, entities in zip(examples, entities):
            example = deepcopy(example)
            if process is not None:
                example = process(example)
            if isinstance(example["text"], str):
                tokens = tokenizer.tokenize(example["text"])
            else:
                tokens = example["text"]
            example["entities"] = []
            for entity in entities:                 # entity
                example["entities"].append([])
                for start, end, label in entity:    # span
                    example["entities"][-1].append(
                        (start, end, label, tokens[start: end])
                    )
            updated.append(example)
        return updated

    if opts.use_sinusoidal_width_embedding:
        logger.info("Initializing sinusoidal width embedding")
        def _init_weight(out: nn.Parameter):
            """
            Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
            the 2nd half of the vector. [dim // 2:]
            """
            n_pos, dim = out.shape
            position_enc = np.array(
                [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
            )
            out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
            sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
            out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            out.detach_()
            return out
        model.head.width_embedding.weight = _init_weight(model.head.width_embedding.weight)

    # trainer
    logger.info("initializing trainer")
    metrics = [
        SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-N"]}, "micro", entity_type="all"),
        SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-N"]}, "micro", entity_type="continuous"),
        SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-N"]}, "micro", entity_type="discontinuous"),
        SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-N"]}, "micro", entity_type="examples_without_discontiguous"),
        SequenceLabelingScoreEntity({label.split("-")[1] for label in \
            data_class.get_labels() if label not in ["O", "II-N"]}, "micro", entity_type="examples_with_discontinuous"),
        SequenceLabelingScoreSpan({label for label in \
            data_class.get_labels() if label not in ["O",]}, average="micro")
    ]
    trainer = Trainer(opts=opts, model=model, metrics=metrics, logger=logger)

    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={"vocab": tokenizer})

    if opts.do_eval:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        if opts.eval_all_checkpoints:
            checkpoints = find_all_checkpoints(checkpoint_dir=opts.output_dir)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.evaluate(dev_data=dev_dataset, save_result=True, save_dir=prefix, multi_thresh=True)

            # 保存为样本，用于分析
            char2token = ProcessConvertLevel(tokenizer, "char2token")
            token2char = ProcessConvertLevel(tokenizer, "token2char")
            results = load_pickle(os.path.join(checkpoint, f"dev_eval_results.pkl"))
            entities = list(chain(*[batch["groundtruths"] for batch in results]))
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[0])
            with open(os.path.join(checkpoint, "groundtruths.txt"), "w") as f:
                for example in examples:
                    text = example["text"]
                    entities = []
                    for entity in example["entities"]:
                        pos = []
                        for start, end, label, _ in entity:
                            pos.extend([start, end - 1])
                        entities.append(",".join(list(map(str, pos))) + " " + entity[0][2])
                    entities = "|".join(entities)
                    f.write(text + "\n")
                    f.write(entities + "\n")
                    f.write("\n")
            with open(os.path.join(checkpoint, "groundtruths.json"), "w") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            entities = list(chain(*[batch["groundtruth_spans"] for batch in results]))
            with open(os.path.join(checkpoint, "groundtruths.span.jsonl"), "w") as f:
                for example, spans in zip(examples, entities):
                    text = example["text"]
                    tokens = tokenizer.tokenize(text)
                    entities_span = []
                    for ((start, start_prefix), (end, end_prefix), label) in spans:
                        string = tokens[start: end]
                        entities_span.append((start, end, f"{start_prefix}{end_prefix}-{label}", string))
                    example["entities"] = entities_span
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

            entities = list(chain(*[batch["predictions"] for batch in results]))
            examples = update_example_entities(tokenizer, dev_dataset.examples, entities, dev_dataset.process_piplines[0])
            with open(os.path.join(checkpoint, "evaluations.txt"), "w") as f:
                for example in examples:
                    text = example["text"]
                    entities = []
                    for entity in example["entities"]:
                        pos = []
                        for start, end, label, _ in entity:
                            pos.extend([start, end - 1])
                        entities.append(",".join(list(map(str, pos))) + " " + entity[0][2])
                    entities = "|".join(entities)
                    f.write(text + "\n")
                    f.write(entities + "\n")
                    f.write("\n")
            with open(os.path.join(checkpoint, "evaluations.json"), "w") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            entities = list(chain(*[batch["predict_spans"] for batch in results]))
            with open(os.path.join(checkpoint, "evaluations.span.jsonl"), "w") as f:
                for example, spans in zip(examples, entities):
                    text = example["text"]
                    tokens = tokenizer.tokenize(text)
                    entities_span = []
                    for ((start, start_prefix), (end, end_prefix), label) in spans:
                        string = tokens[start: end]
                        entities_span.append((start, end, f"{start_prefix}{end_prefix}-{label}", string))
                    example["entities"] = entities_span
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

    if opts.do_predict:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)

            # 保存为样本，用于分析
            results = load_pickle(os.path.join(checkpoint, f"test_predict_results.pkl"))
            entities = list(chain(*[batch["predictions"] for batch in results]))
            examples = update_example_entities(tokenizer, test_dataset.examples, entities, test_dataset.process_piplines[0])
            with open(os.path.join(checkpoint, "predictions.json"), "w") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return trainer

if __name__ == "__main__":
    main(build_opts())

