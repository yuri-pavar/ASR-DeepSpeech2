import re
from string import ascii_lowercase
from collections import defaultdict
import torch
import gzip
import os, shutil, wget
import kenlm
from pyctcdecode import build_ctcdecoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
            self, 
            alphabet=None, 
            use_lm=None,
            lm_name=None,
            vocab_name=None,
            **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.lm_name = lm_name
        self.vocab_name = vocab_name

        if use_lm:
            # https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/03_eval_performance.ipynb
            assert self.lm_name is not None and self.vocab_name is not None, "For lm_ctc_beam_search lm_name and vocab_path should be assigned"
            # model download
            lm_gzip_path = self.lm_name + '.arpa.gz'
            if not os.path.exists(lm_gzip_path):
                print('Downloading pruned 3-gram model.')
                lm_url = f'http://www.openslr.org/resources/11/{lm_gzip_path}'
                lm_gzip_path = wget.download(lm_url)
                print('Downloaded the 3-gram language model.')
            else:
                print('Pruned .arpa.gz already exists.')
            # unzip model
            # NOTE: since out nemo vocabulary is all lowercased, we need to convert all librispeech data as well
            uppercase_lm_path = self.lm_name + '.arpa'
            if not os.path.exists(uppercase_lm_path):
                with gzip.open(lm_gzip_path, 'rb') as f_zipped:
                    with open(uppercase_lm_path, 'wb') as f_unzipped:
                        shutil.copyfileobj(f_zipped, f_unzipped)
                print('Unzipped the 3-gram language model.')
            else:
                print('Unzipped .arpa already exists.')
            lowercase_lm_path = f'lowercase_{self.lm_name}.arpa'
            if not os.path.exists(lowercase_lm_path):
                with open(uppercase_lm_path, 'r') as f_upper:
                    with open(lowercase_lm_path, 'w') as f_lower:
                        for line in f_upper:
                            f_lower.write(line.lower())
            print('Converted language model file to lowercase.')
            if not os.path.exists(lm_gzip_path):
                # vocab download
                lm_vocab_path = wget.download(f'http://www.openslr.org/resources/11/{self.vocab_name}.txt')
            # load unigram list
            with open(f"{self.vocab_name}.txt") as f:
                unigram_list = [t.lower() for t in f.read().strip().split("\n")]
            # load kenlm Model
            # https://github.com/kensho-technologies/pyctcdecode/issues/58
            # kenlm_model = kenlm.Model(lowercase_lm_path)
            self.decoder = build_ctcdecoder(
                [self.EMPTY_TOK] + list(self.alphabet),
                # kenlm_model,
                lowercase_lm_path,
                unigram_list,
                alpha=0.7,
                beta=3.0,
            )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        # pass  # TODO
        decoded = []
        last_char_ind = 0
        for ind in inds:
            if last_char_ind == ind:
                continue
            if ind != 0:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return ''.join(decoded)
    
    # def _expand_and_merge_path(self, dp, next_token_probs, ind2char):
    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            # cur_char = ind2char[ind]
            cur_char = self.ind2char[ind]
            # print(f'---- {type(dp)}')
            # print(f'---- {dp}')
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
        new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp
    
    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x [1]) [:beam_size])
    
    # def ctc_beam_search(self, probs, beam_size, ind2char):
    def ctc_beam_search(self, probs, beam_size):
        dp = {
            ('', self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            # dp = self._expand_and_merge_path(dp, prob, ind2char)
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, beam_size)
        dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])] 
        # return dp
        return dp[0][
                0
            ]  # dp[0] - это лучшие (prefix, proba), а dp[0][0] - соответственно лучший префикс

    def lm_ctc_beam_search(self, probs, beam_size):
        return self.decoder.decode(logits=probs, beam_width=beam_size)



    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
