# `minitorch.tokenizers`

Two tokenizer implementations behind a common `Tokenizer` interface:
`CharTokenizer` (character-level, used by the training notebook) and
`BPETokenizer` (byte-pair-encoding, word-level subword merges).

```python
Tokenizer                      # ABC: encode(text) -> list[int], decode(ids) -> str
  TOKEN_UNKNOWN = "<UNK>"
  TOKEN_EOT = "<EOT>"
  TOKEN_SOT = "<SOT>"

CharTokenizer()
  .build_vocab(corpus: list[str])
  .encode(text) -> list[int]   # [<SOT>, ...chars..., <EOT>]
  .decode(ids) -> str

BPETokenizer(vocab_size=10000)
  .train(corpus: list[str])
  .encode(text) -> list[int]
  .encode_batch(texts) -> list[list[int]]
  .decode(ids) -> str
```

## `CharTokenizer`
Builds a vocabulary of every unique lowercase character seen across the
corpus, plus `<SOT>`/`<UNK>`/`<EOT>`. `encode` wraps every string with
`<SOT> ... <EOT>` and maps unseen characters to `<UNK>`.

## `BPETokenizer`
Classic greedy BPE: pre-tokenizes into words/punctuation
(`re.findall(r'\w+|[^\w\s]', text)`), starts from a per-character vocabulary
(tagging each word's last character with `<EOT>` so merges respect word
boundaries), then repeatedly finds the most frequent adjacent pair
(`_count_byte_pairs`) and merges it (`_merge_pair`) until `vocab_size` is
reached or no pairs remain.

## Usage
```python
from minitorch.tokenizers import CharTokenizer

tok = CharTokenizer()
tok.build_vocab([text])
ids = tok.encode(text)
tok.decode(ids) == f"<sot>{text.lower()}<eot>"   # roughly, modulo unknowns
```
