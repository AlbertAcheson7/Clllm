import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class TokenSupConLoss(nn.Module):
    """
    Token-Level 监督对比学习损失函数。
    拉近同类 token（例如都是 B-Disease）的表征距离，拉远不同类 token 的表征距离。
    """
    def __init__(self, temperature=0.1):
        super(TokenSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        计算给定特征和标签的监督对比损失。
        Args:
            features: 投影后的特征矩阵，形状通常为 [N, proj_dim]，其中 N = batch_size * seq_len (展平后的所有有效 token 数量)。
            labels: 对应的标签，形状通常为 [N]，每个元素代表对应 token 的真实类别 ID。
        Returns:
            loss: 标量张量 (Scalar Tensor)，计算得到的监督对比损失值。
        """
        device = features.device
        
        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # labels掩码
        labels = labels.contiguous().view(-1, 1)
        mask_labels = torch.eq(labels, labels.T).float().to(device)
        
        # 消除对角线(自身)
        logits_mask = torch.ones_like(mask_labels) - torch.eye(mask_labels.shape[0], device=device)
        mask_labels = mask_labels * logits_mask
        
        max_logits, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - max_logits.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_sum = mask_labels.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask_labels * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos
        
        valid_loss = loss[mask_labels.sum(1) > 0]
        if len(valid_loss) > 0:
            return valid_loss.mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class ContrastiveNERModel(nn.Module):
    def __init__(self, model_name, num_labels, proj_dim=128, dropout_prob=0.1):
        super(ContrastiveNERModel, self).__init__()
        # 主干网络（BioBERT）
        self.encoder = AutoModel.from_pretrained(model_name)  # 
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # 投影头(Projector)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.encoder.config.hidden_size, proj_dim)
        )
        
        # 分类头
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask): # input_ids: [B, L], attention_mask: [B, L]
        """
        模型的前向传播。
        Args:
            input_ids: 形状为 [B, L]，输入序列的 Token ID 张量。
            attention_mask: 形状为 [B, L]，注意力掩码张量（1表示真实 token，0表示 padding padding）。
        Returns:
            logits: 形状为 [B, L, num_labels]，用于常规交叉熵分类预测的未归一化分数。
            proj_features: 形状为 [B, L, proj_dim]，经过投影头降维后的特征，后续将展平送入 TokenSupConLoss 进行对比学习。
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # outputs.last_hidden_state: [B, L, hidden_size]
        sequence_output = outputs.last_hidden_state 
        
        sequence_output_dropped = self.dropout(sequence_output)
        
        proj_features = self.projector(sequence_output_dropped)
        logits = self.classifier(sequence_output_dropped)
        
        return logits, proj_features
