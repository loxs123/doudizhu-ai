import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def masked_log_softmax_sparse(data, mask, chosen_action):
    """
    稀疏 masked softmax
    data: [B, T, A] logits
    mask: [B, T, A] 0/1 合法性
    chosen_action: [B, T] 动作索引
    返回: [B, T] chosen_action 对应 log_softmax 值
    """
    B, T, A = data.shape
    flat_data = data.view(-1, A)
    flat_mask = mask.view(-1, A)
    flat_chosen = chosen_action.view(-1)

    chosen_log_probs = []
    for i in range(flat_data.shape[0]):
        valid_idx = flat_mask[i].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            # 没有合法动作，返回 -inf
            chosen_log_probs.append(torch.tensor(float('-inf'), device=data.device))
            continue

        valid_logits = flat_data[i, valid_idx]
        valid_log_probs = F.log_softmax(valid_logits, dim=-1)

        match_pos = (valid_idx == flat_chosen[i]).nonzero(as_tuple=True)
        if match_pos[0].numel() == 0:
            # chosen_action 非法，返回 -inf
            chosen_log_probs.append(torch.tensor(float('-inf'), device=data.device))
        else:
            pos = match_pos[0].item()
            chosen_log_probs.append(valid_log_probs[pos])

    return torch.stack(chosen_log_probs, dim=0).view(B, T)



B, T, A = 2, 3, 1000
data = torch.randn(B, T, A)
mask = (torch.rand(B, T, A) < 0.1).int()  # 10% 合法
chosen_action = torch.randint(0, A, (B, T))

logp = masked_log_softmax_sparse(data, mask, chosen_action)
print(logp.shape)  # [B, T]
