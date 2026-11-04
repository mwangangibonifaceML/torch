import re

from collections import Counter

class Tokenizer:
    """
    Base Tokenizer providing the basic architecture that tokenizers follow.
    
    It provides the following architecture
    - encode(): convert text to token ID.
    - decode(): convert token IDs back to text.
    """
    
    #* predefined tokens
    TOKEN_UNKNOWN= "<UNK>"
    TOKEN_EOT = "<EOT>"
    TOKEN_SOT = "<SOT>"
    
    def encode(self, text: str) -> list[int]:
        """
        Convert text to token IDs.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """
        raise NotImplementedError(
            f"encode() not implemented on Tokenizer class.\n"
            f" ❌ Called encode() on abstract base class {self.__class__.__name__}\n"
            f" 🔦 Tokenizer class is just an interface. Implement encode() in concrete class such as CharTokenizer() or BPETokennizer().\n"
            f" ✅ Example: tokenizer = CharTokenizer(); token_ids = tokenizer.encode('hello world')")
        
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token ids back to raw text

        Args:
            token_ids (list[int]): IDs to convert to text

        Returns:
            str: A string of text obtained from the token Ids
        """
        raise NotImplementedError(
            f"decode() not implemented on Tokenizer class.\n"
            f" ❌ Called decode() on abstract base class {self.__class__.__name__}\n"
            f" 🔦 Tokenizer class is just an interface. Implement decode() in concrete class such as CharTokenizer() or BPETokennizer().\n"
            f" ✅ Example: tokenizer = CharTokenizer(); text = tokenizer.decode([1, 2, 3])")
    

class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that converts text into token IDs based on individual characters.
    
    This tokenizer creates a vocabulary of unique characters from the input text and
    assigns a unique ID to each character. It also includes special tokens for unknown
    characters and end-of-text.
    
    Character tokenization provides a simple, robust foundation for text processing. 
    The key insight is that with a small vocabulary (typically <100 characters),
    we can represent any text without unknown tokens.

    **Trade-offs**:
    - **Pro**: No out-of-vocabulary issues, handles any language
    - **Con**: Long sequences (1 char = 1 token), limited semantic understanding
    - **Use case**: When robustness is more important than efficiency
    """
    
    def __init__(self) ->None:
        self.vocabulary = []
        
        #* create bidirectional mappings (empty at first)
        self.char_to_id: dict[str, int] = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char: dict[int, str] = {idx: char for idx, char in enumerate(self.vocabulary)}
        self.next_id: int = int(len(self.vocabulary))
        
        #* Add special tokens to the vocabulary
        self._add_token(self.TOKEN_SOT)
        self._add_token(self.TOKEN_UNKNOWN)
        self._add_token(self.TOKEN_EOT)

        #* vocabulary builder flag
        self.is_vocab_built = False
        self.vocabulary = self.char_to_id.keys()
        
    def _add_token(self, token: str) -> None:
        """
        Add a token to the vocabulary.
        
        Args:
            token (str): The token to be added to the vocabulary.
        """
        if token not in self.char_to_id:
            self.char_to_id[token] = self.next_id
            self.id_to_char[self.next_id] = token
            self.next_id += 1
            
        self.vocab_size: int = len(self.char_to_id)
    
    def build_vocab(self, corpus: list[str] | None) -> None:
        """
        Build the vocabulary from a list of text samples.
        
        Args:
            corpus (list[str]): A list of text samples to build the vocabulary from.
        """
        #* lower case and strip whitespace from each text sample
        all_chars = set() #* ensures that there are no duplicates
        
        if not corpus:
            raise ValueError(
                """Corpus cannot be empty. Please provide a valid corpus for building the vocabulary."""
            )
            
        corpus = [text.lower().strip() for text in corpus]
        
        #* Iterate through each text sample and add each character to the vocabulary
        for text in corpus:
            all_chars.update(text)
            
            for char in sorted(all_chars):
                self._add_token(char)
                
        #* update vocab builder flag
        self.is_vocab_built = True
                
    def encode(self, text: str) -> list[int]:
        """
        Convert text to token IDs based on the character-level vocabulary.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """
        if not self.is_vocab_built:
            raise RuntimeError(
                'Vocabulary not built. call build_vocab() first.'
            )
            
        token_ids: list[int] = [self.char_to_id[self.TOKEN_SOT]]
        
        for word in text.lower():
            for char in word:
                token_id = self.char_to_id.get(char, self.char_to_id[self.TOKEN_UNKNOWN])
                token_ids.append(token_id)
                
        token_ids.append(self.char_to_id[self.TOKEN_EOT])
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back to raw text based on the character-level vocabulary.
        
        Args:
            token_ids (list[int]): A list of token IDs to be converted back to text.
        
        Returns:
            str: A string of text obtained from the token IDs.
        """
        chars: list[str] = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, self.TOKEN_UNKNOWN)
            chars.append(char)
        return ''.join(chars)
    
class BPETokenizer(Tokenizer):

    """
    Byte Pair Encoding (BPE) that learns subwords units.
    
    Starts by with character level vocabulary then learns
    the most frequent character pairs and merges them together
    to form single tokens.
    
    Keeps repeating until desired vocabulary size is attained.

    """
    def __init__(self, vocab_size: int = 10000):
        if not isinstance(vocab_size, int) or vocab_size < 0:
            raise ValueError("vocab size must be a positive integer.")
        self.vocab:          list = []
        self.tokens_to_ids:  dict[str, int] = {}
        self.ids_to_tokens:  dict[int, str] = {}
        self.merges:         list[tuple[str, str]] = []
        
        #* training flag
        self.is_trained = False
        
    def pre_tokenize(self, text: str) -> list[str]:
        """
        Splits text into words and punctuation while preserving 
        essential boundaries.
        """
        #* split the text into words and punctuations.
        return re.findall(r'\w+|[^\w\s]', text)
        
    def _get_word_tokens(self, word: str) -> list[str]:
        """
        Get the individual tokens of a word

        Args:
            word (str): The word to get tokens from

        Returns:
            list[str]: Individual tokens from a word
        """
        tokens = list(word)
        if len(tokens) == 0:
            return []
        tokens[-1] += Tokenizer.TOKEN_EOT
        return tokens
    
    
    def train(self, corpus: list[str]):
        """
        Train the BPE to learn pairs and merge them.
        
        It initializes character vocabulary and run a greedy merge loop
        using _count_byte_pairs to find the best pair and _merge_pairs()
        to merge them

        Args:
            corpus (list[str]): Document to learn and train from (expected to be a list of sentences)
            vocab_size (int): maximum vocabulary size allowed
        """
        full_corpus = []
        
        if corpus is None or len(corpus) == 0:
            raise ValueError("Corpus cannot be empty. Please provide a valid corpus for training.")
        
        for sentence in corpus:
            full_corpus.extend(self.pre_tokenize(sentence.lower()))

        # if vocab_size:
        #     self.vocab_size = vocab_size
            
        #* count word occurences in the corpus to get the frequency of each word
        word_freq: Counter = Counter(full_corpus)
        word_tokens: dict[str, list[str]] = {}
        vocab = set()
        
        #* get the words tokens from corpus
        for word in full_corpus:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)
        
        #* update the vocabulary using the tokens and
        #* unknown token    
        self.vocab = sorted(list(vocab))
        
        if Tokenizer.TOKEN_UNKNOWN not in self.vocab:
            self.vocab.insert(0, Tokenizer.TOKEN_UNKNOWN)
            
        #* find the best pair(s) and merge them
        while len(self.vocab) < self.vocab_size:
            pair_counts = self._count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break
            
            best_pair = pair_counts.most_common(1)[0][0]
            new_token = self._merge_pair(word_tokens, best_pair)
    
            self.merges.append(best_pair)
            self.vocab.append(new_token)
            
        self._build_mapping()
        
        self.is_trained = True
        
    def _build_mapping(self):
        """
        Create token to id and id to token mappings
        """
        self.tokens_to_ids = {token: id for id, token in enumerate(self.vocab)}
        self.ids_to_tokens = {id: token for id, token in enumerate(self.vocab)}
            
    def _count_byte_pairs(self, word_token: dict[str, list[str]], word_count: Counter)-> Counter:
        """
        Count the frequency of all adjacent token pairs in a corpus
        
        Each pair count is weighted by the frequency of word containing it
        in the corpus, so most frequent words contribute more the statistic

        Args:
            word_token (dict[str, list[str]]): dictionary with word and its tokens
            word_count (Counter): dictionary showing the frequency count of the word

        Returns:
            Counter: Pair count freqeuncies of the adjacent tokens
        """
        pair_count = Counter()
        for word, count in word_count.items():
            tokens = word_token[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_count[pair] += count
                
        return pair_count
                
    def _merge_pair(self, word_token: dict[str, list[str]], pair: tuple[str, str]) -> str:
        """
        Merge the most frequent pair in all word token lists.

        Scans through every word's tokens and replaces adjacent occurrences
        of the pair with a single concatenated token. Modifies word_tokens
        in place and returns the new merged token string.
        """
        merged_pair = pair[0] + pair[1]
        
        for word in word_token:
            tokens = word_token[word]
            new_tokens: list[str] = []
            counter = 0
            
            
            while counter < len(tokens):
                if (counter < len(tokens) - 1) and tokens[counter] == pair[0]\
                    and tokens[counter + 1] == pair[1]:
                    new_tokens.append(merged_pair)
                    counter += 2
                else:
                    new_tokens.append(tokens[counter])
                    counter += 1
            word_token[word] = new_tokens
                    
        return merged_pair
    
    def _apply_merges(self, tokens: list[str]) -> list[str]:
        if not self.merges:
            return tokens
        
        current_tokens = list(tokens)
        for pair in self.merges:
            new_tokens = []
            i = 0
            
            while (i < len(current_tokens)):
                if (i < len(current_tokens) - 1) and\
                current_tokens[i] == pair[0] and \
                current_tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
                    
            current_tokens = new_tokens
        return current_tokens
    
    def encode(self, text: str)-> list[int]:
        """
        Encode a sentence into token ids

        Args:
            text (str): sentence to encode

        Returns:
            list[int]: a list token ids from the sentence
        """
        if not self.is_trained:
            raise RuntimeError(
                'Tokenizer not trained. call train() first.'
            )
            
        words = self.pre_tokenize(text.lower())
        all_tokens = []
        
        for word in words:
            tokens = self._get_word_tokens(word)
            merged_tokens = self._apply_merges(tokens)
            all_tokens.extend(merged_tokens)
        
        tokens_ids = []
        for token in all_tokens:
            id = self.tokens_to_ids.get(token,
                self.tokens_to_ids[Tokenizer.TOKEN_UNKNOWN])
            tokens_ids.append(id)
        
        return tokens_ids
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode a batch of text

        Args:
            texts (list[str]): list of sentences to encode

        Returns:
            list[list[int]]: list token ids for each sentence
        """
        batch_ids = []
        for text in texts:
            tokens_ids = self.encode(text)
            # batch_ids.extend(tokens_ids)
            batch_ids.append(tokens_ids)
        return batch_ids
    
    def decode(self, token_ids: list[int]) -> str:
        #* return empty string id token mapping doesn't exist
        if not self.ids_to_tokens:
            return ""
        
        #* iterate through the token ids and get the corresponding
        #* token from id to token mapping
        text = [self.ids_to_tokens.get(t, Tokenizer.TOKEN_UNKNOWN) for t in token_ids]
        
        #* join all the tokens together and perform some clean up
        text = "".join(text)
        text = text.replace(Tokenizer.TOKEN_EOT, " ")
        text = " ".join(text.split())
        return text
    
    @property
    def vocab_size(self):
        "Actual vocab size."
        return len(self.vocab)
    
    
    
if __name__ == "__main__":
    tokenizer = CharTokenizer()
    corpus = [
        "My name is Boniface Mwangangi.",
        "Am not that cool but I like writing code",
        "and most of all if it's ML code"
    ]
    tokenizer.build_vocab(corpus)
    text = "My name is Boniface"
    token_ids = tokenizer.encode(text)
    print(token_ids)
    decoded_output = tokenizer.decode(token_ids)
    print(decoded_output)
    print("".join(("<SOT>", text, "<EOT>")))