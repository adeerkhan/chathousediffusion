import torch
import torch.nn as nn

# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的位置编码表
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        # x: (sequence_length, batch_size, d_model)
        # 将位置编码添加到输入的特征上
        return x + self.pe[:x.size(0), :]


# Transformer模型
class TransformerModule(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, d_model, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.d_model = d_model
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        # 编码层，将输入特征转换为期望的维度
        self.encoder = nn.Linear(num_features, d_model)
        # Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 解码层，将Transformer输出转换回原始特征维度
        self.decoder = nn.Linear(d_model, num_features)
        self.init_weights()

    def init_weights(self):
        # 初始化权重以改善小模型训练的收敛性
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)
        # 调整x以适配Transformer输入要求的维度
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, num_features)
        x = self.encoder(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x.permute(1, 0, 2)  # 调回(batch_size, sequence_length, num_features)


# 示例使用
if __name__ == "__main__":
    num_features = 128
    sequence_length = 10
    batch_size = 32
    model = TransformerModule(num_features=num_features, num_heads=8, num_layers=6, d_model=512)
    input_tensor = torch.randn(batch_size, sequence_length, num_features)
    output = model(input_tensor)
    print(output.shape)  # (batch_size, sequence_length, num_features)

