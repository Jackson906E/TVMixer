# import torch
# import torch.nn as nn
# from layers.Autoformer_EncDec import series_decomp
# from layers.Embed import DataEmbedding_wo_pos
# from layers.StandardNorm import Normalize
# from layers.SelfAttention_Family import FullAttention, AttentionLayer


# class DFT_series_decomp(nn.Module):
#     def __init__(self, top_k=5):
#         super(DFT_series_decomp, self).__init__()
#         self.top_k = top_k

#     def forward(self, x):
#         xf = torch.fft.rfft(x)
#         freq = abs(xf)
#         freq[0] = 0
#         top_k_freq, top_list = torch.topk(freq, self.top_k)
#         xf[freq <= top_k_freq.min()] = 0
#         x_season = torch.fft.irfft(xf)
#         x_trend = x - x_season
#         return x_season, x_trend

# #1

# # class VariateCrossAttention(nn.Module):
# #     def __init__(self, configs):
# #         super(VariateCrossAttention, self).__init__()
# #         self.variate_attention = AttentionLayer(
# #             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
# #             configs.d_model, configs.n_heads
# #         )

# #     def forward(self, x, attn_mask=None):
# #         # 在每个时间步上应用Cross Attention，使不同变量在同一时间步上相互交互
# #         B, T, D = x.shape
# #         x = x.permute(1, 0, 2)  # 转换为 [T, B, D] 以应用每个时间步的交互
# #         variate_out, _ = self.variate_attention(x, x, x, attn_mask=attn_mask)
# #         return variate_out.permute(1, 0, 2)  # 转回 [B, T, D]


# #2

# # class VariateCrossAttention(nn.Module):
# #     def __init__(self, configs, window_size=24):
# #         super(VariateCrossAttention, self).__init__()
# #         self.variate_attention = AttentionLayer(
# #             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
# #             configs.d_model, configs.n_heads // 2
# #         )
# #         self.window_size = window_size 

# #     def forward(self, x, attn_mask=None):
# #         B, T, D = x.shape
# #         output = torch.zeros_like(x)
        
# #         for i in range(0, T, self.window_size):
# #             end = min(i + self.window_size, T)
# #             window_x = x[:, i:end, :]
# #             window_x = window_x.permute(1, 0, 2)  # 转换为 [窗口大小, B, D]
            
            
# #             variate_out, _ = self.variate_attention(window_x, window_x, window_x, attn_mask=attn_mask)
# #             variate_out = variate_out.permute(1, 0, 2)  # 转回 [B, 窗口大小, D]
            
           
# #             output[:, i:end, :] = variate_out
            
# #         return output


# #3

# # class PositionalEncoding(nn.Module):
# #     def __init__(self, d_model, max_len=5000):
# #         super(PositionalEncoding, self).__init__()
# #         pe = torch.zeros(max_len, d_model)
# #         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# #         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
# #         pe[:, 0::2] = torch.sin(position * div_term)
# #         pe[:, 1::2] = torch.cos(position * div_term)
# #         pe = pe.unsqueeze(0).transpose(0, 1)
# #         self.register_buffer('pe', pe)

# #     def forward(self, x):
# #         x = x + self.pe[:x.size(0), :]
# #         return x


# # class VariateCrossAttentionWithPosEncoding(nn.Module):
# #     def __init__(self, configs):
# #         super(VariateCrossAttentionWithPosEncoding, self).__init__()
# #         self.variate_attention = AttentionLayer(
# #             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
# #             configs.d_model, configs.n_heads
# #         )
# #         self.positional_encoding = PositionalEncoding(configs.d_model)

# #     def forward(self, x, attn_mask=None):
# #         B, T, D = x.shape
# #         x = self.positional_encoding(x.permute(1, 0, 2))  # [T, B, D]
# #         variate_out, _ = self.variate_attention(x, x, x, attn_mask=attn_mask)
        
# #         return variate_out.permute(1, 0, 2)  # [B, T, D]


# #4

# # class DynamicGateVariateCrossAttention(nn.Module):
# #     def __init__(self, configs):
# #         super(DynamicGateVariateCrossAttention, self).__init__()
# #         self.variate_attention = AttentionLayer(
# #             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
# #             configs.d_model, configs.n_heads
# #         )
# #         # Dynamic gating mechanism to control cross-variate attention per time step
# #         self.gating_layer = nn.Sequential(
# #             nn.Linear(configs.d_model, configs.d_model),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x, attn_mask=None):
# #         B, T, D = x.shape
# #         x = x.permute(1, 0, 2)  # [T, B, D]
        
# #         # Compute gating values
# #         gating_values = self.gating_layer(x)  # [T, B, D]
        
# #         # Apply gating to variate-wise attention
# #         gated_x = x * gating_values
# #         variate_out, _ = self.variate_attention(gated_x, gated_x, gated_x, attn_mask=attn_mask)
        
# #         return variate_out.permute(1, 0, 2)  # [B, T, D]

# class DynamicGateVariateCrossAttention(nn.Module):
#     def __init__(self, configs, window_size=5):
#         super(DynamicGateVariateCrossAttention, self).__init__()
#         self.variate_attention = AttentionLayer(
#             FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#             configs.d_model, configs.n_heads // 2
#         )
#         # Dynamic gating mechanism to control cross-variate attention per window
#         self.gating_layer = nn.Sequential(
#             nn.Linear(configs.d_model, configs.d_model),
#             nn.Sigmoid()
#         )
#         self.window_size = window_size  # Define the window size for segmenting the sequence

#     def forward(self, x, attn_mask=None):
#         B, T, D = x.shape
#         output = torch.zeros_like(x)  # Initialize the output tensor

#         for i in range(0, T, self.window_size):
#             end = min(i + self.window_size, T)
#             window_x = x[:, i:end, :]  # Extract the current window
#             window_x = window_x.permute(1, 0, 2)  # Transform to [window_size, B, D] for attention

#             # Compute dynamic gating values for the current window
#             gating_values = self.gating_layer(window_x)  # [window_size, B, D]

#             # Apply gating to the attention input
#             gated_x = window_x * gating_values

#             # Perform variate-wise cross-attention within the window
#             variate_out, _ = self.variate_attention(gated_x, gated_x, gated_x, attn_mask=attn_mask)

#             # Transform back to [B, window_size, D] and store the result in the output tensor
#             output[:, i:end, :] = variate_out.permute(1, 0, 2)

#         return output

    

# class MultiScaleSeasonMixing(nn.Module):
#     def __init__(self, configs):
#         super(MultiScaleSeasonMixing, self).__init__()
#         self.down_sampling_layers = torch.nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                     ),
#                     nn.GELU(),
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                     ),
#                 )
#                 for i in range(configs.down_sampling_layers)
#             ]
#         )

#     def forward(self, season_list):
#         out_high = season_list[0]
#         out_low = season_list[1]
#         out_season_list = [out_high.permute(0, 2, 1)]

#         for i in range(len(season_list) - 1):
#             out_low_res = self.down_sampling_layers[i](out_high)
#             out_low = out_low + out_low_res
#             out_high = out_low
#             if i + 2 <= len(season_list) - 1:
#                 out_low = season_list[i + 2]
#             out_season_list.append(out_high.permute(0, 2, 1))

#         return out_season_list

# class MultiScaleTrendMixing(nn.Module):
#     def __init__(self, configs):
#         super(MultiScaleTrendMixing, self).__init__()
#         self.up_sampling_layers = torch.nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                     ),
#                     nn.GELU(),
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                     ),
#                 )
#                 for i in reversed(range(configs.down_sampling_layers))
#             ])

#     def forward(self, trend_list):
#         trend_list_reverse = trend_list.copy()
#         trend_list_reverse.reverse()
#         out_low = trend_list_reverse[0]
#         out_high = trend_list_reverse[1]
#         out_trend_list = [out_low.permute(0, 2, 1)]

#         for i in range(len(trend_list_reverse) - 1):
#             out_high_res = self.up_sampling_layers[i](out_low)
#             out_high = out_high + out_high_res
#             out_low = out_high
#             if i + 2 <= len(trend_list_reverse) - 1:
#                 out_high = trend_list_reverse[i + 2]
#             out_trend_list.append(out_low.permute(0, 2, 1))

#         out_trend_list.reverse()
#         return out_trend_list
    
    


# class PastDecomposableMixing(nn.Module):
#     def __init__(self, configs, window_size=5):
#         super(PastDecomposableMixing, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.down_sampling_window = configs.down_sampling_window
#         self.layer_norm = nn.LayerNorm(configs.d_model)
#         self.dropout = nn.Dropout(configs.dropout)
#         self.channel_independence = configs.channel_independence

#         if configs.decomp_method == 'moving_avg':
#             self.decomposition = series_decomp(configs.moving_avg)
#         elif configs.decomp_method == "dft_decomp":
#             self.decomposition = DFT_series_decomp(configs.top_k)
#         else:
#             raise ValueError('Decomposition method error')

#         if configs.channel_independence == 0:
#             self.cross_layer = nn.Sequential(
#                 nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
#                 nn.GELU(),
#                 nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
#             )

#         self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
#         self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

#         # 新增的 VariateCrossAttention 层
#         # self.variate_cross_attention = VariateCrossAttention(configs, window_size = window_size)
#         self.variate_cross_attention = DynamicGateVariateCrossAttention(configs,window_size = window_size)

#         self.out_cross_layer = nn.Sequential(
#             nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
#             nn.GELU(),
#             nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
#         )

#     def forward(self, x_list):
#         length_list = []
#         for x in x_list:
#             _, T, _ = x.size()
#             length_list.append(T)

#         season_list = []
#         trend_list = []
#         for x in x_list:
#             season, trend = self.decomposition(x)
#             if self.channel_independence == 0:
#                 season = self.cross_layer(season)
#                 trend = self.cross_layer(trend)
#             season_list.append(season.permute(0, 2, 1))
#             trend_list.append(trend.permute(0, 2, 1))

#         out_season_list = self.mixing_multi_scale_season(season_list)
#         out_trend_list = self.mixing_multi_scale_trend(trend_list)

#         out_list = []
#         for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
#             combined = out_season + out_trend
            
#             combined = self.variate_cross_attention(combined)
            
#             if self.channel_independence:
#                 combined = ori + self.out_cross_layer(combined)
#             out_list.append(combined[:, :length, :])
#         return out_list

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.configs = configs
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.down_sampling_window = configs.down_sampling_window
#         self.channel_independence = configs.channel_independence
#         self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
#                                          for _ in range(configs.e_layers)])

#         self.preprocess = series_decomp(configs.moving_avg)
#         self.enc_in = configs.enc_in

#         if self.channel_independence == 1:
#             self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
#         else:
#             self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

#         self.layer = configs.e_layers
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.predict_layers = torch.nn.ModuleList(
#                 [
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                         configs.pred_len,
#                     )
#                     for i in range(configs.down_sampling_layers + 1)
#                 ]
#             )

#             if self.channel_independence == 1:
#                 self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
#             else:
#                 self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
#                 self.out_res_layers = torch.nn.ModuleList([
#                     nn.Linear(
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                     )
#                     for i in range(configs.down_sampling_layers + 1)
#                 ])
#                 self.regression_layers = torch.nn.ModuleList(
#                     [
#                         nn.Linear(
#                             configs.seq_len // (configs.down_sampling_window ** i),
#                             configs.pred_len,
#                         )
#                         for i in range(configs.down_sampling_layers + 1)
#                     ]
#                 )

#             self.normalize_layers = torch.nn.ModuleList(
#                 [
#                     Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
#                     for i in range(configs.down_sampling_layers + 1)
#                 ]
#             )

#     def out_projection(self, dec_out, i, out_res):
#         dec_out = self.projection_layer(dec_out)
#         out_res = out_res.permute(0, 2, 1)
#         out_res = self.out_res_layers[i](out_res)
#         out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
#         dec_out = dec_out + out_res
#         return dec_out

#     def pre_enc(self, x_list):
#         if self.channel_independence == 1:
#             return (x_list, None)
#         else:
#             out1_list = []
#             out2_list = []
#             for x in x_list:
#                 x_1, x_2 = self.preprocess(x)
#                 out1_list.append(x_1)
#                 out2_list.append(x_2)
#             return (out1_list, out2_list)
        
    
#     def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
#         if self.configs.down_sampling_method == 'max':
#             down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
#         elif self.configs.down_sampling_method == 'avg':
#             down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
#         elif self.configs.down_sampling_method == 'conv':
#             padding = 1 if torch.__version__ >= '1.5.0' else 2
#             down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
#                                   kernel_size=3, padding=padding,
#                                   stride=self.configs.down_sampling_window,
#                                   padding_mode='circular',
#                                   bias=False)
#         else:
#             return x_enc, x_mark_enc
#         x_enc = x_enc.permute(0, 2, 1)
#         x_enc_ori = x_enc
#         x_mark_enc_mark_ori = x_mark_enc
#         x_enc_sampling_list = []
#         x_mark_sampling_list = []
#         x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
#         x_mark_sampling_list.append(x_mark_enc)

#         for i in range(self.configs.down_sampling_layers):
#             x_enc_sampling = down_pool(x_enc_ori)
#             x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
#             x_enc_ori = x_enc_sampling
#             if x_mark_enc is not None:
#                 x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
#                 x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

#         x_enc = x_enc_sampling_list
#         x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

#         return x_enc, x_mark_enc

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
#         x_list = []
#         x_mark_list = []
#         if x_mark_enc is not None:
#             for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
#                 B, T, N = x.size()
#                 x = self.normalize_layers[i](x, 'norm')
#                 if self.channel_independence == 1:
#                     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
#                 x_list.append(x)
#                 x_mark = x_mark.repeat(N, 1, 1)
#                 x_mark_list.append(x_mark)
#         else:
#             for i, x in zip(range(len(x_enc)), x_enc, ):
#                 B, T, N = x.size()
#                 x = self.normalize_layers[i](x, 'norm')
#                 if self.channel_independence == 1:
#                     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
#                 x_list.append(x)

#         enc_out_list = []
#         x_list = self.pre_enc(x_list)
#         if x_mark_enc is not None:
#             for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
#                 enc_out = self.enc_embedding(x, x_mark)
#                 enc_out_list.append(enc_out)
#         else:
#             for i, x in zip(range(len(x_list[0])), x_list[0]):
#                 enc_out = self.enc_embedding(x, None)
#                 enc_out_list.append(enc_out)

#         for i in range(self.layer):
#             enc_out_list = self.pdm_blocks[i](enc_out_list)

#         dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
#         dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
#         dec_out = self.normalize_layers[0](dec_out, 'denorm')
#         return dec_out

#     def future_multi_mixing(self, B, enc_out_list, x_list):
#         dec_out_list = []
#         if self.channel_independence == 1:
#             x_list = x_list[0]
#             for i, enc_out in zip(range(len(x_list)), enc_out_list):
#                 dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
#                 dec_out = self.projection_layer(dec_out)
#                 dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
#                 dec_out_list.append(dec_out)

#         else:
#             for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
#                 dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
#                 dec_out = self.out_projection(dec_out, i, out_res)
#                 dec_out_list.append(dec_out)

#         return dec_out_list

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out_list
#         else:
#             raise ValueError('Only forecast tasks implemented yet')

    











import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


# class CNNVariateMixing(nn.Module):
#     def __init__(self, configs, window_size=24):
#         super(CNNVariateMixing, self).__init__()
#         # Using Conv1d with groups to allow for independent channel-wise filtering
#         self.conv_layer = nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1, groups=configs.d_model)
#         self.window_size = window_size
#         self.gating_layer = nn.Sequential(
#             nn.Linear(configs.d_model, configs.d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         B, T, D = x.shape
#         output = torch.zeros_like(x)  # Initialize the output tensor [B, T, D]

#         for i in range(0, T, self.window_size):
#             end = min(i + self.window_size, T)
#             window_x = x[:, i:end, :]  # Extract the current window [B, window_size, D]
            
#             # Permute to apply Conv1d, which expects [B, D, window_size]
#             window_x = window_x.permute(0, 2, 1)  # [B, D, window_size]
#             cnn_out = self.conv_layer(window_x)  # Apply CNN, output shape [B, D, window_size]

#             # Apply gating after permuting back to [B, window_size, D]
#             gating_values = self.gating_layer(cnn_out.permute(0, 2, 1))  # [B, window_size, D]
#             gated_out = cnn_out.permute(0, 2, 1) * gating_values  # Element-wise multiply [B, window_size, D]

#             # Store the gated output back into the output tensor
#             output[:, i:end, :] = gated_out

#         return output

class CNNVariateMixing(nn.Module):
    def __init__(self, configs, window_size=10):
        super(CNNVariateMixing, self).__init__()
        # Using Conv1d without groups to enable cross-variate interaction
        self.conv_layer = nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1)
        self.window_size = window_size
        self.gating_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, D = x.shape
        output = torch.zeros_like(x)  # Initialize the output tensor [B, T, D]

        for i in range(0, T, self.window_size):
            end = min(i + self.window_size, T)
            window_x = x[:, i:end, :] 
            
            # Permute to apply Conv1d across variates for each window
            window_x = window_x.permute(0, 2, 1)  # [B, D, window_size]
            cnn_out = self.conv_layer(window_x)  

            # Apply gating after permuting back to [B, window_size, D]
            gating_values = self.gating_layer(cnn_out.permute(0, 2, 1))  
            gated_out = cnn_out.permute(0, 2, 1) * gating_values  

            # Store the gated output back into the output tensor
            output[:, i:end, :] = gated_out

        return output



class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs, window_size=24):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decomposition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('Decomposition method error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # CNN-based variate mixing
        self.variate_mixing = CNNVariateMixing(configs, window_size)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            combined = out_season + out_trend
            combined = self.variate_mixing(combined)
            if self.channel_independence:
                combined = ori + self.out_cross_layer(combined)
            out_list.append(combined[:, :length, :])
        return out_list



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.out_res_layers = torch.nn.ModuleList([
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])
                self.regression_layers = torch.nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)
        
    
    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')

    