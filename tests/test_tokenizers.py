import pytest
import numpy as np

from minitorch.tensor import Tensor
from minitorch.tokenizers import Tokenizer, BPETokenizer, CharTokenizer

def test_unit_base_tokenizer():
    """🧪 Test base tokenizer interface."""
    print("🧪 Unit Test: Base Tokenizer Interface...")

    #* Test that base class defines the interface
    tokenizer = Tokenizer()

    #* Should raise NotImplementedError for both methods
    with pytest.raises(NotImplementedError):
        tokenizer.encode("test")
        
    with pytest.raises(NotImplementedError):
        tokenizer.decode([1,2,3,4])

    print("✅ Base tokenizer interface works correctly!")
    

def test_unit_char_tokenizer():
    """🧪 Test character tokenizer implementation."""
    print("🧪 Unit Test: Character Tokenizer...")

    #* Test basic functionality
    tokenizer = CharTokenizer()

    #* Test vocabulary setup
    assert tokenizer.vocab_size == 3  # 3 chars ---> (SOT, UNK, EOT)
    assert tokenizer.id_to_char[0] == Tokenizer.TOKEN_SOT
    assert tokenizer.id_to_char[1] == Tokenizer.TOKEN_UNKNOWN
    assert tokenizer.id_to_char[2] == Tokenizer.TOKEN_EOT
    assert '<EOT>' in tokenizer.char_to_id
    assert '<SOT>' in tokenizer.char_to_id
    assert '<UNK>' in tokenizer.char_to_id
    assert tokenizer.is_vocab_built is False
    
    #* Test encoding
    text = "hello"
    tokenizer.build_vocab([text])
    tokens = tokenizer.encode(text)
    expected = [0,4,3,5,5,6,2]  # h,e,l,l,o (based on actual vocab order)
    assert tokens == expected, f"Expected {expected}, got {tokens}"

    #* Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == "".join(("<SOT>",text,"<EOT>")), f"Expected '{text}', got '{decoded}'"

    #* Test unknown character handling
    tokens_with_unk = tokenizer.encode("hello!")
    assert tokens_with_unk[-2] == 1  # '!' should map to <UNK>

    # Test vocabulary building
    # Expected behavior: Invoking build_vocab overwrites the vocabulary passed
    # in __init__ above, so after this the 'w' should no longer be a token
    corpus = ["hola mundo", "test text"]
    tokenizer.build_vocab(corpus)
    assert 'w' not in tokenizer.char_to_id
    assert 't' in tokenizer.char_to_id
    assert 'x' in tokenizer.char_to_id

    print("✅ Character tokenizer works correctly!")
    
def test_unit_bpe_tokenizer():
    """🧪 Test BPE tokenizer implementation."""
    print("🧪 Unit Test: BPE Tokenizer...")

    # Test basic functionality with simple corpus
    corpus = ["hello", "world", "hello", "hell"]  # "hell" and "hello" share prefix
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # Check that vocabulary was built
    assert len(tokenizer.vocab) > 0
    assert Tokenizer.TOKEN_UNKNOWN in tokenizer.vocab

    # Test helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith(Tokenizer.TOKEN_EOT), "Should have end-of-word marker"


    # Test encoding/decoding
    text = "hello"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)

    # Test round-trip on training data should work well
    for word in corpus:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        assert decoded == word, "Known words must round-trip exactly"

    print("✅ BPE tokenizer works correctly!")
    
if __name__ == "__main__":
    test_unit_base_tokenizer()
    test_unit_char_tokenizer()
    test_unit_bpe_tokenizer()