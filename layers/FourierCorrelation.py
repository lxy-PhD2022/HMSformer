# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
import math
import pywt
import  copy


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
class ZBlock(nn.Module):
    def __init__(self,seq_len, pred_len, modes=0, mode_select_method='random', b_coeffs=[0.25, 0.25, 0.25, 0.25], a_coeffs=[1, -0.3]):
        super(ZBlock, self).__init__()
        self.b_coeffs = nn.Parameter(torch.tensor(b_coeffs, dtype=torch.float32), requires_grad=True)
        self.a_coeffs = nn.Parameter(torch.tensor(a_coeffs, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        对输入信号x应用滤波器
        :param x: 输入信号（三维张量）
        :return: 滤波后的信号
        """
        # x is expected to be of shape [batch, features, sequence_length]
        batch_size, num_features, seq_length = x.shape
        y = torch.zeros_like(x)
        y_temp = y.clone()
        for i in range(len(self.b_coeffs)):
            y_temp[:, :, i:] = y[:, :, i:] + self.b_coeffs[i] * x[:, :, :seq_length - i]
        y = y_temp

        y_temp = y.clone()
        for j in range(1, len(self.a_coeffs)):
            y_temp[:, :, j:] = y[:, :, j:] - self.a_coeffs[j] * y[:, :, :seq_length - j]
        y = y_temp
        # 应用前馈系数
        # for i in range(len(self.b_coeffs)):
        #     y[:, :, i:] += self.b_coeffs[i] * x[:, :, :seq_length - i]
        #
        # # 应用反馈系数
        # for j in range(1, len(self.a_coeffs)):
        #     y[:, :, j:] -= self.a_coeffs[j] * y[:, :, :seq_length - j]

        return y
class Wavelet(nn.Module):
    def __init__(self, seq_len, pred_len, modes=0, mode_select_method='rndom', wave_type='db1'):
        super(Wavelet, self).__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.seq_len = seq_len

        # self.trend1 = nn.Linear((math.ceil(seq_len / 2) + 2), (math.ceil(seq_len / 2) * 8))
        # self.trend2 = nn.Linear((math.ceil(seq_len / 2) * 8), (math.ceil(seq_len / 2) * 32))
        # self.trend3 = nn.Linear((math.ceil(seq_len / 2) * 32), (math.ceil(seq_len / 2) * 64))
        # self.trend4 = nn.Linear((math.ceil(seq_len / 2) * 64), (math.ceil(seq_len / 2) * 32))
        # self.trend5 = nn.Linear((math.ceil(seq_len / 2) * 32), (math.ceil(seq_len / 2) + 2))
        # self.seasonal1 = nn.Linear((math.ceil(seq_len / 2) + 2), (math.ceil(seq_len / 2) * 8))
        # self.seasonal2 = nn.Linear((math.ceil(seq_len / 2) * 8), (math.ceil(seq_len / 2) * 32))
        # self.seasonal3 = nn.Linear((math.ceil(seq_len / 2) * 32), (math.ceil(seq_len / 2) * 64))
        # self.seasonal4 = nn.Linear((math.ceil(seq_len / 2) * 64), (math.ceil(seq_len / 2) * 32))
        # self.seasonal5 = nn.Linear((math.ceil(seq_len / 2) * 32), (math.ceil(seq_len / 2) + 2))
        # self.changeback = nn.Linear(2 * seq_len, seq_len)
        self.wave = wave_type

        # 封装成函数
    def sgn(self, num):
        if (num > 0.0):
            return 1.0
        elif (num == 0.0):
            return 0.0
        else:
            return -1.0

    def wavelet_noising(self, new_df):
        data = new_df
        data = data.T.tolist()  # 将np.ndarray()转为列表

        w = pywt.Wavelet('dB1')  # 选择dB10小波基
        ca3, cd3, cd2, cd1 = pywt.wavedec(data, w, level=3)  # 3层小波分解
        ca3 = ca3  # ndarray数组减维：(1，a)->(a,)
        cd3 = cd3
        cd2 = cd2
        cd1 = cd1
        length1 = len(cd1)
        length0 = len(data)

        abs_cd1 = np.abs(np.array(cd1))
        median_cd1 = np.median(abs_cd1)

        sigma = (1.0 / 0.6745) * median_cd1
        lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
        usecoeffs = []
        usecoeffs.append(ca3)

        # 软阈值方法
        for k in range(length1):
            if (abs(cd1[k]) >= lamda / np.log2(2)):
                cd1[k] = self.sgn(cd1[k]) * (abs(cd1[k]) - lamda / np.log2(2))
            else:
                cd1[k] = 0.0

        length2 = len(cd2)
        for k in range(length2):
            if (abs(cd2[k]) >= lamda / np.log2(3)):
                cd2[k] = self.sgn(cd2[k]) * (abs(cd2[k]) - lamda / np.log2(3))
            else:
                cd2[k] = 0.0

        length3 = len(cd3)
        for k in range(length3):
            if (abs(cd3[k]) >= lamda / np.log2(4)):
                cd3[k] = self.sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
            else:
                cd3[k] = 0.0

        usecoeffs.append(cd3)
        usecoeffs.append(cd2)
        usecoeffs.append(cd1)
        recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构
        return recoeffs

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        wavelet = self.wave
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] = torch.Tensor(self.wavelet_noising(x[i][j].cpu().numpy()))
        # print(Yl.shape)
        # Yl = self.trend1(Yl)
        # Yl = self.trend2(Yl)
        # Yl = self.trend3(Yl)
        # Yl = self.trend4(Yl)
        # Yl = self.trend5(Yl)
        # Yh1 = self.seasonal1(Yh[0])
        # Yh1 = self.seasonal2(Yh1)
        # Yh1 = self.seasonal3(Yh1)
        # Yh1 = self.seasonal4(Yh1)
        # Yh[0] = self.seasonal5(Yh1)
        # print(np.array(ca).shape)
        # print(np.array(cd).shape)
        # remove ca
        # ca.fill(0)
        # x = self.changeback(x)
        # x = x.squeeze(-2)
        # print("wave:", x.shape)
        return x
# ########## fourier layer for my network #############
class Kalman(nn.Module):
    def __init__(self, seq_len, pred_len, modes=0, mode_select_method='random', frequency=50):
        super(Kalman, self).__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.Linear_fourier  = nn.Linear(((seq_len//2)+1)*2, ((seq_len//2)+1)*2)
        self.sampling_rate = 100
        self.cutoff_frequency = frequency


    def forward(self, x):

        x_ft = torch.fft.rfft(x, dim=-1)
        # 创建频率掩码用于高通滤波
        freq = torch.fft.rfftfreq(x.size(-1), d=1 / self.sampling_rate)
        filter_mask = torch.abs(freq) < self.cutoff_frequency

        # 应用高通滤波器
        x_ft_filtered = x_ft * filter_mask.to(x_ft.device).unsqueeze(0).unsqueeze(0)

        x_ft_flat = torch.view_as_real(x_ft_filtered).view(x.size(0) * x.size(1), -1)
        # Perform Fourier neural operations
        out_ft = self.Linear_fourier(x_ft_flat)

        out_ft = out_ft.view(x.size(0) * x.size(1), -1, 2)
        out_ft = torch.view_as_complex(out_ft).view(x.size(0), x.size(1), -1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x


# class FourierBlock(nn.Module):
#     def __init__(self, seq_len, pred_len, modes=0, mode_select_method='random', frequency=50, sampling_rate=100):
#         super(FourierBlock, self).__init__()
#         """
#         1D Fourier block with multiple CNN layers.
#         """
#         self.seq_len = seq_len
#         fft_size = ((seq_len // 2) + 1) * 2
#         self.sampling_rate = sampling_rate
#         self.cutoff_frequency = frequency
#
#         # Define multiple CNN layers
#         self.conv1d_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, padding_mode='replicate')
#         self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, padding_mode='replicate')
#         self.conv1d_3 = nn.Conv1d(32, 1, kernel_size=1)  # Reduce to the original dimension
#
#     def forward(self, x):
#         x_ft = torch.fft.rfft(x, dim=-1)
#         # High-pass filter
#         # freq = torch.fft.rfftfreq(x.size(-1), d=1 / self.sampling_rate)
#         # filter_mask = torch.abs(freq) < self.cutoff_frequency
#         # x_ft_filtered = x_ft * filter_mask.to(x_ft.device).unsqueeze(0).unsqueeze(0)
#
#         x_ft_flat = torch.view_as_real(x_ft).view(x.size(0) * x.size(1), 1, -1)
#         # Apply multiple CNN layers
#         x_cnn = F.relu(self.conv1d_1(x_ft_flat))
#         x_cnn = F.relu(self.conv1d_2(x_cnn))
#         x_cnn = self.conv1d_3(x_cnn)  # Final layer to match the original dimension
#
#         x_cnn = x_cnn.view(x.size(0) * x.size(1), -1, 2)
#         x_cnn = torch.view_as_complex(x_cnn).view(x.size(0), x.size(1), -1)
#
#         x = torch.fft.irfft(x_cnn, n=x.size(-1))
#
#         return x
class FourierBlock(nn.Module):
    def __init__(self, seq_len, pred_len, modes=0, mode_select_method='random', frequency=50, sampling_rate=100):
        super(FourierBlock, self).__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain,
        it does FFT, linear transform, and Inverse FFT.
        """
        self.Linear_fourier  = nn.Linear(((seq_len//2)+1)*2, ((seq_len//2)+1)*8)
        self.Linear_fourier2 = nn.Linear(((seq_len // 2) + 1) * 8, ((seq_len // 2) + 1) * 2)
        self.sampling_rate = sampling_rate
        self.cutoff_frequency = frequency


    def forward(self, x):

        x_ft = torch.fft.rfft(x, dim=-1)
        # 创建频率掩码用于高通滤波
        freq = torch.fft.rfftfreq(x.size(-1), d=1 / self.sampling_rate)
        filter_mask = torch.abs(freq) < self.cutoff_frequency

        # 应用di通滤波器
        x_ft_filtered = x_ft * filter_mask.to(x_ft.device).unsqueeze(0).unsqueeze(0)

        x_ft_flat = torch.view_as_real(x_ft_filtered).view(x.size(0) * x.size(1), -1)
        # Perform Fourier neural operations
        out_ft = self.Linear_fourier2(self.Linear_fourier(x_ft_flat))

        out_ft = out_ft.view(x.size(0) * x.size(1), -1, 2)
        out_ft = torch.view_as_complex(out_ft).view(x.size(0), x.size(1), -1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x

# # ########## fourier layer #############
# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
#         super(FourierBlock, self).__init__()
#         print('fourier enhanced block used!')
#         """
#         1D Fourier block. It performs representation learning on frequency domain,
#         it does FFT, linear transform, and Inverse FFT.
#         """
#         # get modes on frequency domain
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
#         print('modes={}, index={}'.format(modes, self.index))
#
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
#
#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#         return torch.einsum("bhi,hio->bho", input, weights)
#
#     def forward(self, q, k, v, mask):
#         # size = [B, L, H, E]
#         B, L, H, E = q.shape
#         x = q.permute(0, 2, 3, 1)
#         # Compute Fourier coefficients
#         x_ft = torch.fft.rfft(x, dim=-1)
#         # Perform Fourier neural operations
#         out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         for wi, i in enumerate(self.index):
#             if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
#                 continue
#             out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
#         # Return to time domain
#         x = torch.fft.irfft(out_ft, n=x.size(-1))
#         return (x, None)


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)
        
        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)