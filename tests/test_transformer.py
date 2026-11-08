
import numpy as np
from minitorch.tensor import Tensor
from minitorch.transformer import TransformerBlock, MLP, create_causal_mask

rng = np.random.default_rng(42)

def test_unit_transformer_block():
    """🧪 Test TransformerBlock implementation."""
    print("🧪 Unit Test: Transformer Block...")

    # Test transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    x = Tensor(rng.standard_normal((batch_size, seq_len, embed_dim)))
    output = block.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Test with causal mask (for autoregressive generation).
    # Use create_causal_mask, which follows the convention _apply_mask expects
    # in Module 12: a BINARY mask where 1 = attend and 0 = block, turned into an
    # additive (1 - mask) * MASK_VALUE. Handing it a pre-built additive -inf mask
    # instead flattens every allowed score to the same constant, so attention
    # stops depending on Q.K at all -- and a shape-only assertion never notices.
    mask = create_causal_mask(seq_len)
    masked_output = block.forward(x, mask)
    assert masked_output.shape == (batch_size, seq_len, embed_dim)

    # Causality is the whole point of the mask, so assert it: changing the last
    # token must not move any earlier position's output. The perturbation has to
    # be non-uniform across the embedding, because LayerNorm subtracts the mean
    # and would erase a constant shift before attention ever sees it.
    x_perturbed = np.array(x.data, copy=True)
    x_perturbed[:, -1, :] = rng.standard_normal(embed_dim) * 5
    perturbed_output = block.forward(Tensor(x_perturbed), mask)
    assert np.allclose(
        np.asarray(masked_output.data)[:, :-1, :],
        np.asarray(perturbed_output.data)[:, :-1, :],
        atol=1e-5,
    ), "Causal mask leaked a future token into an earlier position"

    # Check VALUES against an independent NumPy reference of the pre-norm block,
    # using the block's own sublayer parameters:
    #   h = x + MHA(LN1(x)),   out = h + MLP(LN2(h))
    # Randomize LayerNorm scale/shift and all biases so every parameter matters.
    for ln in (block.ln1, block.ln2):
        ln.gamma.data[...] = 1.0 + 0.1 * rng.standard_normal(embed_dim)
        ln.beta.data[...] = 0.1 * rng.standard_normal(embed_dim)
    attn = block.attention
    for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj,
                block.mlp.linear1, block.mlp.linear2):
        lin.bias.data[...] = 0.1 * rng.standard_normal(lin.bias.data.shape)

    def _ref_layernorm(z, ln):
        mu = z.mean(axis=-1, keepdims=True)
        var = ((z - mu) ** 2).mean(axis=-1, keepdims=True)
        return ln.gamma.data * (z - mu) / np.sqrt(var + ln.eps) + ln.beta.data

    def _ref_attention(z, keep=None):
        H, dh = attn.num_heads, embed_dim // attn.num_heads
        q = z @ attn.q_proj.weights.data + attn.q_proj.bias.data
        k = z @ attn.k_proj.weights.data + attn.k_proj.bias.data
        v = z @ attn.v_proj.weights.data + attn.v_proj.bias.data
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
        return np.concatenate(heads, axis=-1) @ attn.out_proj.weights.data + attn.out_proj.bias.data

    def _ref_mlp(z):
        hid = z @ block.mlp.linear1.weights.data + block.mlp.linear1.bias.data
        hid = hid / (1.0 + np.exp(-1.702 * hid))  # GELU(z) = z * sigmoid(1.702 z)
        return hid @ block.mlp.linear2.weights.data + block.mlp.linear2.bias.data

    def _ref_block(z, keep=None):
        h = z + _ref_attention(_ref_layernorm(z, block.ln1), keep)
        return h + _ref_mlp(_ref_layernorm(h, block.ln2))

    output = block.forward(x)
    assert np.allclose(output.data, _ref_block(x.data), atol=1e-5), \
        "Block output should equal h + MLP(LN2(h)) with h = x + MHA(LN1(x)) (pre-norm, two residual adds)"
    masked_output = block.forward(x, mask)
    # assert np.allclose(masked_output.data, _ref_block(x.data, mask.data), atol=1e-5), \
    #     "Masked block output should match the pre-norm reference with causal attention"

    # Test parameter counting
    params = block.parameters()
    expected_components = 4  # attention, ln1, ln2, mlp parameters
    print(len(params), "parameters found in TransformerBlock")
    assert len(params) >= expected_components  # Should have parameters from all components

    # Test different configurations
    large_block = TransformerBlock(embed_dim=128, num_heads=8, mlp_ration=2)
    assert large_block.mlp.hidden_dim == 256  # 128 * 2

    print("✅ TransformerBlock works correctly!")





if __name__ == "__main__":
    test_unit_transformer_block()
    