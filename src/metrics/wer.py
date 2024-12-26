from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
    

class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = log_probs.cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        cers = [
            calc_wer(
                self.text_encoder.normalize_text(text[i]),
                self.text_encoder.ctc_beam_search(predictions[i][:lengths[i]], self.beam_size)
            ) for i in range(len(log_probs))
        ]
        return sum(cers) / len(cers)


class LMWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = log_probs.cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        cers = [
            calc_wer(
                self.text_encoder.normalize_text(text[i]),
                self.text_encoder.lm_ctc_beam_search(predictions[i][:lengths[i]], self.beam_size)
            ) for i in range(len(log_probs))
        ]
        return sum(cers) / len(cers)
