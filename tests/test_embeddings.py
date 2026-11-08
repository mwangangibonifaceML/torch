import math
import numpy as np
import pytest

from minitorch.embeddings import Embedding, PositionalEncoding, EmbeddingLayer
from minitorch.tensor import Tensor


rng = np.random.default_rng(42)


def test_unit_embedding_init():
    """🧪 Test Embedding.__init__ implementation."""
    print("🧪 Unit Test: Embedding.__init__...")

    embed = Embedding(vocab_size=100, embed_dim=64)

    # Check stored attributes
    assert embed.vocab_size == 100, f"Expected vocab_size=100, got {embed.vocab_size}"
    assert embed.embed_dim == 64, f"Expected embed_dim=64, got {embed.embed_dim}"

    # Check weight shape
    assert embed.weight.shape == (100, 64), f"Expected weight shape (100, 64), got {embed.weight.shape}"

    # Check Xavier bounds: limit = sqrt(6 / (100 + 64)) ≈ 0.191
    limit = math.sqrt(6.0 / (100 + 64))
    assert np.all(embed.weight.data >= -limit - 1e-6), "Weights should be >= -limit"
    assert np.all(embed.weight.data <= limit + 1e-6), "Weights should be <= limit"

    print("✅ Embedding.__init__ works correctly!")
    
    
def test_unit_embedding():
    """🧪 Test Embedding layer implementation."""
    print("🧪 Unit Test: Embedding Layer...")

    # Test 1: Basic embedding creation and forward pass
    embed = Embedding(vocab_size=100, embed_dim=64)

    # Single sequence
    tokens = Tensor([1, 2, 3])
    output = embed(tokens)

    assert output.shape == (3, 64), f"Expected shape (3, 64), got {output.shape}"
    assert len(embed.parameters()) == 1, "Should have 1 parameter (weight matrix)"
    assert embed.parameters()[0].shape == (100, 64), "Weight matrix has wrong shape"

    # Test 2: Batch processing
    batch_tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    batch_output = embed(batch_tokens)

    assert batch_output.shape == (2, 3, 64), f"Expected batch shape (2, 3, 64), got {batch_output.shape}"

    # Test 3: Embedding lookup consistency
    single_lookup = embed(Tensor([1]))
    batch_lookup = embed(Tensor([[1]]))

    # Should get same embedding for same token
    assert np.allclose(single_lookup.data[0], batch_lookup.data[0, 0]), "Inconsistent embedding lookup"

    # Test 4: Parameter access
    params = embed.parameters()
    assert len(params) == 1, "Should have 1 parameter"

    print("✅ Embedding layer works correctly!")
    

def test_unit_embedding_backward():
    """🧪 Test Embedding gradient accumulation."""
    print("🧪 Unit Test: Embedding gradients...")

    embed = Embedding(vocab_size=4, embed_dim=2)

    # Token 0 appears twice, token 2 once, tokens 1 and 3 not at all.
    tokens = Tensor([0, 2, 0])
    output = embed(tokens)
    output.sum().backward()

    grad = embed.weight.grad
    assert grad is not None, "No gradient reached embed.weight"

    # Every position contributes a gradient of 1 to its row.
    assert np.allclose(grad[0], [2.0, 2.0]), (
        f"Row 0 is used twice so its gradient should be [2, 2], got {grad[0]}. "
        "Indexed assignment overwrites instead of accumulating; use np.add.at."
    )
    assert np.allclose(grad[2], [1.0, 1.0]), (
        f"Row 2 is used once so its gradient should be [1, 1], got {grad[2]}"
    )
    assert np.allclose(grad[1], [0.0, 0.0]), (
        f"Row 1 is never looked up so its gradient must stay zero, got {grad[1]}"
    )
    assert np.allclose(grad[3], [0.0, 0.0]), (
        f"Row 3 is never looked up so its gradient must stay zero, got {grad[3]}"
    )

    print("✅ Embedding gradients accumulate correctly!")


def test_unit_positional_encoding_init():
    """🧪 Test PositionalEncoding.__init__ implementation."""
    print("🧪 Unit Test: PositionalEncoding.__init__...")

    pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=64)

    # Check stored attributes
    assert pos_enc.max_seq_len == 512, f"Expected max_seq_len=512, got {pos_enc.max_seq_len}"
    assert pos_enc.embed_dim == 64, f"Expected embed_dim=64, got {pos_enc.embed_dim}"

    # Check position embeddings shape
    assert pos_enc.position_embeddings.shape == (512, 64), \
        f"Expected shape (512, 64), got {pos_enc.position_embeddings.shape}"

    # Check values are reasonably small (additive initialization)
    limit = math.sqrt(6.0 / 64)
    assert np.all(pos_enc.position_embeddings.data >= -limit - 1e-6), "Values should be >= -limit"
    assert np.all(pos_enc.position_embeddings.data <= limit + 1e-6), "Values should be <= limit"

    # Check parameters returns the position embeddings
    params = pos_enc.parameters()
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"

    print("✅ PositionalEncoding.__init__ works correctly!")
    
def test_unit_positional_encoding():
    """🧪 Test Positional Encoding implementation."""
    print("🧪 Unit Test: Positional Encoding...")

    # Test 1: Basic functionality
    pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=64)

    # Create sample embeddings
    embeddings = Tensor(rng.standard_normal((2, 10, 64)))
    output = pos_enc(embeddings)

    assert output.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {output.shape}"

    # Test 2: Position consistency
    # Same position should always get same encoding
    emb1 = Tensor(np.zeros((1, 5, 64)))
    emb2 = Tensor(np.zeros((1, 5, 64)))

    out1 = pos_enc(emb1)
    out2 = pos_enc(emb2)

    assert np.allclose(out1.data, out2.data), "Position encodings should be consistent"

    # Test 3: Different positions get different encodings
    short_emb = Tensor(np.zeros((1, 3, 64)))
    long_emb = Tensor(np.zeros((1, 5, 64)))

    short_out = pos_enc(short_emb)
    long_out = pos_enc(long_emb)

    # First 3 positions should match
    assert np.allclose(short_out.data, long_out.data[:, :3, :]), "Position encoding prefix should match"

    # Test 4: Parameters
    params = pos_enc.parameters()
    assert len(params) == 1, "Should have 1 parameter (position embeddings)"
    assert params[0].shape == (512, 64), "Position embedding matrix has wrong shape"

    # Test 5: VALUES. Each position's learned row is added to every batch element:
    #   out[b, t, :] = x[b, t, :] + P[start_pos + t, :]
    P = pos_enc.position_embeddings.data
    expected = embeddings.data + P[np.newaxis, :10, :]
    assert np.allclose(output.data, expected, atol=1e-6), \
        "Output should equal x + position_embeddings[:seq_len] broadcast across the batch"
    assert np.allclose(out1.data[0], P[:5], atol=1e-6), \
        "With zero input, the output should be exactly the first seq_len position rows"

    # start_pos shifts which rows are added (used by the KV cache in Module 18)
    x_step = Tensor(rng.standard_normal((2, 3, 64)))
    out_step = pos_enc.forward(x_step, start_pos=7)
    assert np.allclose(out_step.data, x_step.data + P[np.newaxis, 7:10, :], atol=1e-6), \
        "With start_pos=7 the output should add position rows 7, 8, 9"

    print("✅ Positional encoding works correctly!")
    
    
def test_unit_emblayer_init():
    """🧪 Test EmbeddingLayer.__init__ component assembly."""
    print("🧪 Unit Test: EmbeddingLayer Initialization...")

    # Test 1: Learned PE creates PositionalEncoding
    layer_learned = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding_type='learned')
    assert isinstance(layer_learned.token_embedding, Embedding), "Should create Embedding"
    assert isinstance(layer_learned.pos_encoding, PositionalEncoding), "Should create PositionalEncoding"
    assert len(layer_learned.parameters()) == 2, "Learned PE: 2 param tensors (token + position)"

    # Test 2: Sinusoidal PE creates fixed Tensor
    layer_sin = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding_type='sinusoidal')
    assert isinstance(layer_sin.pos_encoding, Tensor), "Sinusoidal PE should be a Tensor"
    assert len(layer_sin.parameters()) == 1, "Sinusoidal PE: 1 param tensor (token only)"

    # Test 3: None PE stores None
    layer_none = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding_type=None)
    assert layer_none.pos_encoding_type is None, "No PE should store None"
    assert len(layer_none.parameters()) == 1, "No PE: 1 param tensor (token only)"

    # Test 4: Invalid PE raises ValueError
    with pytest.raises(ValueError):
        EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding_type='invalid')

    # Test 5: Configuration stored correctly
    assert layer_learned.vocab_size == 100
    assert layer_learned.embed_dim == 64
    # assert layer_learned.scale_embeddings == False

    print("✅ EmbeddingLayer initialization works correctly!")

    
def test_unit_complete_embedding_system():
    """🧪 Test complete embedding system."""
    print("🧪 Unit Test: Complete Embedding System...")

    # Test 1: Learned positional encoding
    embed_learned = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        pos_encoding_type='learned'
    )

    tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    output_learned = embed_learned(tokens)

    assert output_learned.shape == (2, 3, 64), f"Expected shape (2, 3, 64), got {output_learned.shape}"

    # Test 2: Sinusoidal positional encoding
    embed_sin = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding_type='sinusoidal'
    )

    output_sin = embed_sin(tokens)
    assert output_sin.shape == (2, 3, 64), "Sinusoidal embedding should have same shape"

    # Test 3: No positional encoding
    embed_none = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding_type=None
    )

    output_none = embed_none(tokens)
    assert output_none.shape == (2, 3, 64), "No pos encoding should have same shape"

    # Test 4: 1D input handling
    tokens_1d = Tensor([1, 2, 3])
    output_1d = embed_learned(tokens_1d)

    assert output_1d.shape == (3, 64), f"Expected shape (3, 64) for 1D input, got {output_1d.shape}"
    # Test 6: Parameter counting
    params_learned = embed_learned.parameters()
    params_sin = embed_sin.parameters()
    params_none = embed_none.parameters()

    assert len(params_learned) == 2, "Learned encoding should have 2 parameter tensors"
    assert len(params_sin) == 1, "Sinusoidal encoding should have 1 parameter tensor"
    assert len(params_none) == 1, "No pos encoding should have 1 parameter tensor"

    print("✅ Complete embedding system works correctly!")
    
    
    
if __name__ == "__main__":
    test_unit_embedding_init()
    test_unit_embedding()
    test_unit_embedding_backward()
    test_unit_positional_encoding_init()
    test_unit_positional_encoding()
    test_unit_emblayer_init()
    test_unit_complete_embedding_system()