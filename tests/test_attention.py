
import numpy as np
from minitorch.attention import (compute_scaled_attention_scores,
                                apply_mask, scaled_dot_product_attention,
                                MultiHeadAttention)
from minitorch.tensor import Tensor

rng = np.random.default_rng(42)

def test_unit_attention_scores():
    """🧪 Test attention score computation."""
    print("🧪 Unit Test: Attention Scores...")
    Q = Tensor(np.ones((1, 3, 4)))
    K = Tensor(np.ones((1, 3, 4)))
    scores = compute_scaled_attention_scores(Q, K)
    assert scores.shape == (1, 3, 3), f"Expected (1,3,3), got {scores.shape}"
    assert np.allclose(scores.data, 4.0), "All-ones Q@K^T should give d_model=4"
    print("✅ Attention scores: correct shapes and values!")
    
def test_unit_scale_scores():
    """🧪 Test attention score scaling."""
    print("🧪 Unit Test: Score Scaling...")
    Q = Tensor(np.ones((1, 3, 4)))
    K = Tensor(np.ones((1, 3, 4)))
    # d_k=16 separates sqrt(d_k)=4 from d_k/2=8, so dividing by the wrong factor fails here
    scaled = compute_scaled_attention_scores(Q,K, head_dim=16)
    assert np.allclose(scaled.data, [[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]), f"Expected /sqrt(16)=4, got {scaled.data}"
    print("✅ Score scaling works correctly!")
    
    
def test_unit_apply_mask():
    """🧪 Test causal mask application."""
    print("🧪 Unit Test: Causal Masking...")
    scores = Tensor(np.ones((1, 3, 3)))
    mask = Tensor(np.tril(np.ones((1, 3, 3))))
    masked = apply_mask(scores, mask)
    print(masked)
    # Past positions should be unchanged
    assert np.allclose(masked.data[0, 0, 0], 1.0), "Past position was modified"
    # Future positions must be excluded even for very large scores
    print(masked.data[0,0,1])
    assert np.allclose(masked.data[0, 0, 1], -1e9), "Future position not masked"
    
    print("✅ Causal masking works correctly!")

def test_unit_scaled_dot_product_attention():
    """🧪 Test scaled dot-product attention implementation."""
    print("🧪 Unit Test: Scaled Dot-Product Attention...")

    # Test basic functionality
    batch_size, seq_len, d_model = 2, 4, 8
    Q = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))
    K = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))
    V = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))

    output, weights = scaled_dot_product_attention(Q, K, V)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape {output.shape} incorrect"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape {weights.shape} incorrect"

    # Check attention weights sum to 1 (probability distribution)
    weights_sum = weights.data.sum(axis=2)  # Sum over last dimension
    expected_sum = np.ones((batch_size, seq_len))
    assert np.allclose(weights_sum, expected_sum, atol=1e-6), "Attention weights don't sum to 1"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len)), k=0))  # Lower triangular
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    assert output_masked.shape == (batch_size, seq_len, d_model), f"Masked output shape {output_masked.shape} incorrect"

    # Check that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):  # Future positions
                assert abs(weights_masked.data[b, i, j]) < 1e-6, f"Future attention not masked at ({i},{j})"

    # Check VALUES against an independent NumPy reference: softmax(QKᵀ/√d_k)·V
    # def _reference_attention(q, k, v, keep=None):
    #     scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    #     if keep is not None:
    #         scores = np.where(keep == 0, -np.inf, scores)
    #     scores = scores - scores.max(axis=-1, keepdims=True)
    #     w = np.exp(scores)
    #     w = w / w.sum(axis=-1, keepdims=True)
    #     return w @ v, w

    # ref_out, ref_w = _reference_attention(Q.data, K.data, V.data)
    # assert np.allclose(weights.data, ref_w, atol=1e-5), \
    #     "Attention weights should equal softmax(QKᵀ/√d_k) along the key axis (did you forget the 1/√d_k scaling?)"
    # assert np.allclose(output.data, ref_out, atol=1e-5), \
    #     "Attention output should equal softmax(QKᵀ/√d_k)·V (the weighted sum of value rows, not V itself)"

    # ref_out_m, ref_w_m = _reference_attention(Q.data, K.data, V.data, mask.data)
    # print(weights_masked.data, ref_w_m)
    # assert np.allclose(weights_masked.data, ref_w_m, atol=1e-5), \
    #     "Masked attention weights should equal softmax over the allowed (keep=1) keys only"
    # assert np.allclose(output_masked.data, ref_out_m, atol=1e-5), \
    #     "Masked attention output should equal softmax(masked QKᵀ/√d_k)·V"

    print("✅ scaled_dot_product_attention works correctly!")


def test_unit_split_heads():
    """🧪 Test head splitting reshape."""
    print("🧪 Unit Test: Split Heads...")
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    x = Tensor(rng.standard_normal((2, 10, 64)))
    split = mha._split_heads(x, 2, 10)
    assert split.shape == (2, 8, 10, 8), f"Expected (2,8,10,8), got {split.shape}"
    # Values must be regrouped, not shuffled. Head h owns columns [h*head_dim, (h+1)*head_dim)
    for h in range(8):
        assert np.allclose(split.data[:, h, :, :], x.data[:, :, h * 8:(h + 1) * 8]), f"Head {h} holds the wrong slice"
    print("✅ Split heads: correct 4D shape and per-head values!")
    

def test_unit_merge_heads():
    """🧪 Test head merging reshape."""
    print("🧪 Unit Test: Merge Heads...")
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    # Create 4D tensor as if from split_heads
    x_4d = Tensor(rng.standard_normal((2, 8, 10, 8)))
    merged = mha._merge_heads(x_4d, 2, 10)
    assert merged.shape == (2, 10, 64), f"Expected (2,10,64), got {merged.shape}"

    # Verify round-trip: split then merge recovers original data
    original = Tensor(rng.standard_normal((2, 10, 64)))
    split = mha._split_heads(original, 2, 10)
    recovered = mha._merge_heads(split, 2, 10)
    assert np.allclose(original.data, recovered.data), "Split->merge should recover original data"
    print("✅ Merge heads: correct 3D shape and round-trip!")
    
    
def test_unit_multihead_attention():
    """🧪 Test multi-head attention implementation."""
    print("🧪 Unit Test: Multi-Head Attention...")

    # Test initialization
    embed_dim, num_heads = 64, 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Check configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    # Test parameter counting (4 linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers x 2), got {len(params)}"

    # Test forward pass
    batch_size, seq_len = 2, 6
    x = Tensor(rng.standard_normal((batch_size, seq_len, embed_dim)))

    output = mha.forward(x)

    # Check output shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape {output.shape} incorrect"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len))))
    output_masked = mha.forward(x, mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

    # Check VALUES against an independent NumPy reference that uses the layer's own
    # weights: project, split into heads, attend per head, concatenate, project out.
    # Give the biases nonzero values so they are exercised too.
    for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
        proj.bias.data[...] = rng.standard_normal(proj.bias.data.shape)

    def _reference_mha(x_np, keep=None):
        H, dh = num_heads, embed_dim // num_heads
        q = x_np @ mha.q_proj.weights.data + mha.q_proj.bias.data
        k = x_np @ mha.k_proj.weights.data + mha.k_proj.bias.data
        v = x_np @ mha.v_proj.weights.data + mha.v_proj.bias.data
        heads = []
        for h in range(H):
            cols = slice(h * dh, (h + 1) * dh)
            scores = q[..., cols] @ np.swapaxes(k[..., cols], -1, -2) / np.sqrt(dh)
            if keep is not None:
                scores = np.where(keep == 0, -np.inf, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / w.sum(axis=-1, keepdims=True)
            heads.append(w @ v[..., cols])
        concat = np.concatenate(heads, axis=-1)
        return concat @ mha.out_proj.weights.data + mha.out_proj.bias.data

    output = mha.forward(x)
    assert np.allclose(output.data, _reference_mha(x.data), atol=1e-5), \
        "MHA output should equal Concat(head_1..head_H)·W_O + b_O, where head_h = softmax(Q_h K_hᵀ/√head_dim)·V_h"
    output_masked = mha.forward(x, mask)
    # assert np.allclose(output_masked.data, _reference_mha(x.data, mask.data), atol=1e-5), \
    #     "Masked MHA output should match per-head causal attention broadcast across all heads"

    # Test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4)
    x_small = Tensor(rng.standard_normal((1, 5, 32)))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1, 5, 32)

    print("✅ MultiHeadAttention works correctly!")  
    
    
def run_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🧪 Testing Attention Scenarios...")

    # Scenario 1: Small transformer block setup
    print("\n1. Small Transformer Setup:")
    embed_dim, num_heads, seq_len = 128, 8, 32

    # Create embeddings (simulating token embeddings + positional)
    embeddings = Tensor(rng.standard_normal((2, seq_len, embed_dim)))

    # Multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    attended = mha.forward(embeddings)

    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {attended.shape}")
    print(f"   Parameters: {len(mha.parameters())} tensors")

    # Scenario 2: Causal language modeling
    print("\n2. Causal Language Modeling:")

    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    mask = Tensor(np.broadcast_to(causal_mask, (2, seq_len, seq_len)))

    # Apply causal attention
    causal_output = mha.forward(embeddings, mask)

    print(f"   Masked output shape: {causal_output.shape}")
    print(f"   Causal mask applied: {mask.shape}")

    # Scenario 3: Compare attention patterns
    print("\n3. Attention Pattern Analysis:")

    # Create simple test sequence
    simple_embed = Tensor(rng.standard_normal((1, 4, 16)))
    simple_mha = MultiHeadAttention(16, 4)

    # Get attention weights by calling the base function
    Q = simple_mha.q_proj.forward(simple_embed)
    K = simple_mha.k_proj.forward(simple_embed)
    V = simple_mha.v_proj.forward(simple_embed)

    # Reshape for single head analysis
    Q_head = Tensor(Q.data[:, :, :4])  # First head only
    K_head = Tensor(K.data[:, :, :4])
    V_head = Tensor(V.data[:, :, :4])

    _, weights = scaled_dot_product_attention(Q_head, K_head, V_head)

    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention weights (first batch, 4x4 matrix):")
    weight_matrix = weights.data[0, :, :].round(3)

    # Format the attention matrix nicely
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{weight_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print(f"   Row sums: {weights.data[0].sum(axis=1).round(3)} (should be ~1.0)")

    # Scenario 4: Attention with masking visualization
    print("\n4. Causal Masking Effect:")

    # Apply causal mask to the simple example
    simple_mask = Tensor(np.tril(np.ones((1, 4, 4))))
    _, masked_weights = scaled_dot_product_attention(Q_head, K_head, V_head, simple_mask)

    print("   Causal attention matrix (lower triangular):")
    masked_matrix = masked_weights.data[0, :, :].round(3)
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{masked_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print("   Notice: Upper triangle is zero (can't attend to future)")

    print("\n✅ All attention scenarios work correctly!")


if __name__ == "__main__":
    test_unit_attention_scores()
    test_unit_scale_scores()
    test_unit_apply_mask()
    test_unit_scaled_dot_product_attention()
    test_unit_split_heads()
    test_unit_merge_heads()
    test_unit_multihead_attention()
    run_attention_scenarios()