import torch
import os
import sys
import math
import numpy
import librosa
import torchaudio

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioscore.speaker.models.lstm import LSTMSpeakerEncoder
from audioscore.speaker.config import SpeakerEncoderConfig
from audioscore.speaker.utils.audio import AudioProcessor
from audioscore.speaker.infer import read_json

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        
        # 捷径连接处理维度变化
        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
            self.bn_shortcut = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 处理捷径连接
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn_shortcut(residual)
            
        out += residual
        out = self.relu(out)
        return out

class AudioFeatClassifier_res(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 hidden_size=512, 
                 num_layers=8, 
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)
        
        # 2. 卷积模块（包含3个残差块）
        self.conv_layers = torch.nn.Sequential(
            ResidualBlock(
                in_channels=ppg_dim + vec_dim + pit_embed_dim,
                out_channels=512,
                kernel_size=5,
                stride=4
            ),
            ResidualBlock(512, 512, 5, 2),
            ResidualBlock(512, 512, 5, 2)
        )
        
        # 3. LSTM层（输入维度改为512）
        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, ppg, vec, pit):
        # 维度处理
        pit_emb = self.pit_embed(pit)  # [B, T, 32]
        x = torch.cat([ppg, vec, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        
        # 通过卷积模块（长度压缩约10倍）
        x = self.conv_layers(x)  # [B, 512, T'] (T' ≈ T/10)
        
        # 恢复维度：[B, C, T'] -> [B, T', C]
        x = x.permute(0, 2, 1)

        # print(x.shape)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [B, T', 1024]
        
        # 取最终状态
        last_output = lstm_out[:, -1, :]  # [B, 1024]
        
        # 分类
        return self.classifier(last_output)


class SiameseNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super(SiameseNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 双向RNN编码器（两个子网络共享权重）
        self.rnn = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, 128),  # 双向RNN输出拼接后维度为2*hidden_dim，两个序列拼接后为4*hidden_dim
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 5)  # 输出5个分类
        )
 
    def forward_once(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        output, (hidden, cell) = self.rnn(x)
        
        # 取最后一个时间步的双向隐藏状态
        forward_last = output[:, -1, :self.hidden_dim]
        backward_last = output[:, 0, self.hidden_dim:]
        combined = torch.cat((forward_last, backward_last), dim=1)
        return combined
 
    def forward(self, input1, input2):
        # 编码两个输入序列
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # 计算差异特征
        diff = torch.abs(output1 - output2)
        concat = torch.cat((output1, output2, diff), dim=1)
        
        # 分类
        return self.classifier(concat)
class AudioFeatSiameseNetwork(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear_siam = torch.nn.Linear(self.input_dim, d_model)
        self.enc_siam = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.tf_out = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0,
                    mask):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = self.linear_siam(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.pos_encoder(x)
        x = self.enc_siam(x, src_key_padding_mask=mask)
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0, mask)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0, mask)

        x = x1 - x0

        x = self.tf_out(x, src_key_padding_mask=mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatSiameseNetwork")

        print("saved model to", path)

class AudioFeatSiameseNetwork_cross(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model)
        self.linear_siam = torch.nn.Linear(self.input_dim, d_model)
        self.enc_siam = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.tf_out = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        self.cross_attn = torch.nn.MultiheadAttention(d_model, 16, batch_first=True)
        self.cross_linear = torch.nn.Linear(d_model*2, d_model)

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0,
                    mask):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = self.linear_siam(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.pos_encoder(x)
        x = self.enc_siam(x, src_key_padding_mask=mask)
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0, mask)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0, mask)

        # 交叉注意力机制（双向）
        # x0作为query，x1作为key和value
        attn_output_x0, _ = self.cross_attn(x0, x1, x1, key_padding_mask=mask)
        # x1作为query，x0作为key和value
        attn_output_x1, _ = self.cross_attn(x1, x0, x0, key_padding_mask=mask)
        
        # 合并双向注意力结果
        x = torch.cat([attn_output_x0, attn_output_x1], dim=-1)
        x = self.cross_linear(x)

        x = self.tf_out(x, src_key_padding_mask=mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatSiameseNetwork")

        print("saved model to", path)
        

class AudioFeatCmpNetwork(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=6)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0)

        x = torch.cat([x0, x1], dim=-1)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork")

        print("saved model to", path)

class AudioFeatCmpNetwork_rl(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=163840):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=12)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0)

        x = torch.cat([x0, x1], dim=-1)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork_rl")

        print("saved model to", path)

class AudioFeatCmpNetwork_music(torch.nn.Module):
    def __init__(self, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(1280, 128)

        self.input_dim = 384

        self.overtone_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(9),  # 添加输入LayerNorm
            torch.nn.Linear(9, 128),
            torch.nn.LayerNorm(128),  # 添加线性层后LayerNorm
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),  # 添加第二个线性层后LayerNorm
            torch.nn.LeakyReLU(),
        )
        self.volume_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(1),  # 添加输入LayerNorm
            torch.nn.Linear(1, 128),
            torch.nn.LayerNorm(128),  # 添加线性层后LayerNorm
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),  # 添加第二个线性层后LayerNorm
            torch.nn.LeakyReLU(),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=3)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
    
    def normalize_loud(self, tensor):
        return (tensor+40)/40

    def batch_wise_normalization(self, audio_feats):
        # 沿特征维度(dim=1)计算最大值，keepdim=True保持维度
        max_vals = audio_feats.max(dim=1, keepdim=True).values
        
        # 防止除以0（添加极小值）
        max_vals = max_vals.clamp(min=1e-8)  # 或使用 torch.where(max_vals==0, torch.ones_like(max_vals)*1e-8, max_vals)
        
        # 逐元素除法（广播机制自动对齐维度）
        normalized = audio_feats / max_vals
        return normalized

    def forward_one(self, 
                    audio_feats_overtone,
                    audio_feats_f0,
                    audio_feats_volume):
        # print(audio_feats_f0.max())
        pit_emb = self.pit_embed(audio_feats_f0)
        overtone = self.batch_wise_normalization(audio_feats_overtone)
        overtone = self.overtone_encoder(overtone)
        volume = self.normalize_loud(audio_feats_volume)
        volume = self.volume_encoder(volume.view(volume.shape[0], volume.shape[1], 1))
        x = torch.cat([pit_emb, overtone, volume], dim=-1)
        return x
    
    def forward(self, 
                audio_feats_0_overtone,
                audio_feats_0_f0,
                audio_feats_0_volume,
                audio_feats_1_overtone,
                audio_feats_1_f0,
                audio_feats_1_volume,
                mask):
        x0 = self.forward_one(audio_feats_0_overtone, audio_feats_0_f0, audio_feats_0_volume)
        x1 = self.forward_one(audio_feats_1_overtone, audio_feats_1_f0, audio_feats_1_volume)

        x = torch.cat([x0, x1], dim=-1)
        # print(x.shape)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork_music")

        print("saved model to", path)

class MertFeatCmpNetwork(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,    # MERT hidden size
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)

        # 输入维度 = mert_dim + pit_embed_dim
        self.input_dim = mert_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="relu",
                batch_first=True
            ),
            num_layers=6
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )


    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                mask=None):

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # 分类
        x = self.classifier(x)
        return x

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork")
        print("saved model to", path)

class MuqFeatCmpNetwork(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,    # MERT hidden size
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)

        # 输入维度 = mert_dim + pit_embed_dim
        self.input_dim = mert_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )


    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                mask=None):

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # 分类
        x = self.classifier(x)
        return x

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork")
        print("saved model to", path)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 定义模型（已修改为包含对抗训练）
class MuqFeatCmpNetwork_GRL(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000,
                 speaker_embed_dim=256):
        super().__init__()

        self.input_dim = mert_dim
        
        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        
        # 说话人识别头（对抗训练用）
        self.speaker_discriminator = torch.nn.Sequential(
            torch.nn.Linear(d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, speaker_embed_dim * 2)  # 预测两个说话人的embedding
        )

    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                speaker_embs_0=None,  # 可选，训练时需要
                speaker_embs_1=None,  # 可选，训练时需要
                mask=None,
                alpha=1.0):           # GRL的权重系数

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        
        # 主任务分类
        main_output = self.classifier(encoded)
        
        # 如果提供了说话人embedding，则进行对抗训练
        if speaker_embs_0 is not None and speaker_embs_1 is not None:
            # 使用序列的平均表示
            if mask is not None:
                # 处理mask，计算有效长度
                # print("encoded", encoded.shape)
                # print("mask", mask.shape)
                lengths = mask.sum(dim=1)
                # print("lengths", lengths.shape)
                pooled = (encoded*mask.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])).sum(dim=1)
                # print("pooled", pooled.shape)
                pooled = pooled / lengths.unsqueeze(1)
                # print("pooled", pooled.shape)
            else:
                pooled = encoded.mean(dim=1)
            
            # 应用梯度反转层
            reversed_pooled = GradientReversalLayer.apply(pooled, alpha)
            
            # 说话人识别
            speaker_pred = self.speaker_discriminator(reversed_pooled)
            speaker_pred_0 = speaker_pred[:, :256]  # 预测的第一个说话人embedding
            speaker_pred_1 = speaker_pred[:, 256:]  # 预测的第二个说话人embedding
            
            return main_output, speaker_pred_0, speaker_pred_1
        
        return main_output

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork_GRL")
        print("saved model to", path)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SpkEncoderHelper(torch.nn.Module):
    def __init__(self, root_path=None):
        super(SpkEncoderHelper, self).__init__()
        # python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker
        # model
        print("ROOT_DIR", ROOT_DIR)
        self.model_path =  os.path.join(ROOT_DIR, os.path.join("pretrain", "speaker_pretrain", "best_model.pth.tar"))
        self.config_path = os.path.join(ROOT_DIR, os.path.join("pretrain", "speaker_pretrain", "config.json"))
        if root_path:
            self.model_path = os.path.join(root_path, self.model_path)
            self.config_path = os.path.join(root_path, self.config_path)
        # config
        self.config_dict = read_json(self.config_path)

        # model
        self.config = SpeakerEncoderConfig(self.config_dict)
        self.config.from_dict(self.config_dict)

        self.speaker_encoder = LSTMSpeakerEncoder(
            self.config.model_params["input_dim"],
            self.config.model_params["proj_dim"],
            self.config.model_params["lstm_dim"],
            self.config.model_params["num_lstm_layers"],
        )
        self.use_cuda = True
        self.speaker_encoder.load_checkpoint(
            self.model_path, eval=True, use_cuda=self.use_cuda
        )
        # preprocess
        self.speaker_encoder_ap = AudioProcessor(**self.config.audio)
        # normalize the input audio level and trim silences
        self.speaker_encoder_ap.do_sound_norm = True
        self.speaker_encoder_ap.do_trim_silence = True
    
    def forward(self, wav_list: list[torch.tensor], sr: int, infer: bool = True):
        """
        Args:
            wav_list: list of torch.tensor, 每个元素是一段单声道波形
            sr: int, 输入音频的采样率
            infer: bool, 是否推理模式
        
        Returns:
            torch.Tensor, shape = (len(wav_list), proj_dim)
        """
        device = next(self.speaker_encoder.parameters()).device
        embeds = torch.zeros(len(wav_list), self.speaker_encoder.proj_dim, device=device)

        for i, waveform in enumerate(wav_list):
            # 确保是float32
            waveform = waveform.to(torch.float32).numpy()
            # 自动重采样到模型所需采样率
            target_sr = self.speaker_encoder_ap.sample_rate
            if sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                # waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

            # 提取mel谱
            spec = self.speaker_encoder_ap.melspectrogram(waveform)
            spec = torch.from_numpy(spec.T).unsqueeze(0).to(device).view(1, -1, 80)

            # 计算嵌入
            embed = self.speaker_encoder.compute_embedding(spec, infer=infer)
            embeds[i] = embed

            # print("embed.shape", embed.shape)

        return embeds

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=8000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
 
    def forward(self, x):
        # print("pe.shape", self.pe.shape, x.shape)
        return x + self.pe[:, :x.size(1)].expand(x.size(0), -1, -1)
 
class AudioFeatClassifier_tf(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)
        
        # 2. CNN降维模块
        self.cnn = torch.nn.Sequential(
            # 输入维度: [batch, 1568, seq_len]
            torch.nn.Conv1d(ppg_dim + vec_dim + pit_embed_dim, d_model, 
                          kernel_size=5, stride=2, padding=2),  # 100 -> 20
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv1d(d_model, d_model, 
                          kernel_size=5, stride=2, padding=2),  # 20 -> 4
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv1d(d_model, d_model, 
                          kernel_size=3, stride=2, padding=1)   # 4 -> 2
        )
        
        # 3. Transformer编码器
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=2048,
            dropout=0.3,
            activation='relu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=16)
        
        # 4. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
 
    def forward(self, ppg, vec, pit):
        # 1. 处理pitch特征
        pit_emb = self.pit_embed(pit)  # [batch, seq_len, 32]
        
        # 2. 特征拼接
        x = torch.cat([ppg, vec, pit_emb], dim=-1)  # [batch, 100, 1568]
        
        # 3. CNN降维
        x = x.permute(0, 2, 1)          # [batch, 1568, 100]
        x = self.cnn(x)                 # [batch, 512, 2]
        x = x.permute(0, 2, 1)          # [batch, 2, 512]
        
        # print("x.shape",x.shape)

        # 4. 添加位置编码
        x = self.pos_encoder(x)
        
        # 5. Transformer编码
        x = self.transformer(x)         # [batch, 2, 512]
        
        # 6. 分类（取最后一个时间步）
        x = x[:, -1, :]                 # [batch, 512]
        logits = self.classifier(x)     # [batch, num_classes]
        
        return logits
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), save_path+"/model_weight_tf.pt")
    
    def load_ckpt(self, load_path):
        self.load_state_dict(torch.load(load_path+"/model_weight_tf.pt"))