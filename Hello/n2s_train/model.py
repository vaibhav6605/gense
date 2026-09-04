from transformers import GPT2Config, GPT2LMHeadModel


def build_gpt2(n_units=500, hidden_size=256, num_layers=6, num_heads=8, max_positions=2048):
    shift_num = 3
    vocab_size = n_units + shift_num
    conf = GPT2Config(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=max_positions,
        n_ctx=max_positions,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(conf)
