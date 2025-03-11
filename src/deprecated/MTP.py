# old torch MTP code

class MTPModule(nn.Module):
    """Multi-Token Prediction Module."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.vocab_size = config['vocab_size']
        # No shared embedding here, it's handled in the main Transformer
        self.transformer_block = Block(0)  # Assuming 'mlp' type, adjust if needed
        self.proj_matrix = nn.Linear(self.n_embd, self.n_embd * 2, bias=False) # for combine
        self.rm_f = nn.RMSNorm(self.n_embd)
        # No shared lm_head, handled in main Transformer
        
    def forward(self, h_prev, next_token_emb):
        """
        Args:
            h_prev: Hidden state from the previous depth (or main model). [B, T, C]
            next_token_emb: Embedding of the (i+k)th token. [B, T, C]

        Returns:
            logits for the next token, and hidden state output by mtp module transformer block
        """
        B, T, C = h_prev.shape

        # Combine h_prev and next_token_emb
        combined_input = self.proj_matrix(torch.cat([nn.functional.rms_norm(h_prev, weight=torch.ones(h_prev.shape[-1], device = h_prev.device), eps=1e-5), nn.functional.rms_norm(next_token_emb, weight=torch.ones(next_token_emb.shape[-1], device=next_token_emb.device), eps=1e-5)], dim=-1))

        # Transformer Block
        h_next, _ = self.transformer_block(combined_input) # don't need routing weights
        h_next = self.rm_f(h_next)

        return h_next
