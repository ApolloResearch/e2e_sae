import torch
import torch.nn.functional as F

from sparsify.scripts.analysis.get_acts import Acts, get_acts, tokenizer


def print_info_from_prompt_pos(acts: Acts, batch_idx: int, seqpos: int):
    print(f"Batch {batch_idx}")
    print(f"Sequence Position {seqpos}")
    print(f"Prompt: ...{tokenizer.decode(acts.tokens[batch_idx, :seqpos+1])}")
    print(f"Next: |{tokenizer.decode(acts.tokens[batch_idx, seqpos+1])}|")
    print(f"KL: {acts.kl[batch_idx, seqpos].item():.2f}")
    d = torch.dist(acts.orig[batch_idx, seqpos], acts.recon[batch_idx, seqpos])
    print(f"L2 dist: {d.item():.2f}")
    print("Orig predictions:")
    probs = F.softmax(acts.orig_logits[batch_idx, seqpos], dim=-1)
    for i in acts.orig_logits[batch_idx, seqpos].topk(5).indices:
        print(f"   Prob {probs[i]:.2%} |{tokenizer.decode(i)}|")
    print("Recon predictions:")
    probs = F.softmax(acts.new_logits[batch_idx, seqpos], dim=-1)
    for i in acts.new_logits[batch_idx, seqpos].topk(5).indices:
        print(f"   Prob {probs[i]:.2%} |{tokenizer.decode(i)}|")


def print_info_for_cond(acts: Acts, cond: torch.Tensor, num_samples: int = 10):
    print(f"Condition active on {cond.sum().item()} tokens ({cond.float().mean().item():.2%})")
    rand_idx = torch.randperm(cond.sum().item())[:num_samples]
    for i, (b, s) in enumerate(cond.nonzero()[rand_idx]):
        print(f"\n\n--- Random Sample {i}---")
        print_info_from_prompt_pos(acts, b, s)


acts = get_acts(batches=15)
print_info_for_cond(acts, acts.kl > 1, num_samples=100)
