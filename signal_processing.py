import numpy as np
from scipy.signal import get_window
import scipy.signal as sig


def get_window_and_correction(N, window_type='rect', window_params=None):
    """
    生成窗函数并计算能量补偿系数。
    """
    window_type = window_type.lower()

    # 1. 矩形窗 (全透光)
    if window_type in ['rect', 'boxcar']:
        w = np.ones(N)

    # 2. 图基窗 Tukey (带 alpha 参数的可调平顶帐篷)
    elif window_type == 'tukey':
        # 如果用户传了参数，就用用户的；如果没传或者传了 None，默认给 0.5
        alpha = float(window_params) if window_params is not None else 0.5

        # 这里的关键：必须明确调用 sig.windows.tukey 并传入 alpha
        w = sig.windows.tukey(N, alpha=alpha)

        print(f"  [底层引擎] 已生成 Tukey 窗，当前 alpha = {alpha}")

    # 3. 高斯窗 Gaussian (带标准差参数)
    elif window_type == 'gaussian':
        std = float(window_params) if window_params is not None else 2.5
        w = sig.windows.gaussian(N, std=std)

    # 4. 其他普通窗函数 (Hann, Hamming, Blackman, Flattop 等)
    else:
        try:
            w = sig.get_window(window_type, N)
        except ValueError:
            print(f"  [警告] 未知窗函数 '{window_type}'，自动回退使用矩形窗 (Rect)。")
            w = np.ones(N)

    # 计算能量补偿系数 (相干增益的倒数)
    mean_w = np.mean(w)
    correction = 1.0 / mean_w if mean_w > 0 else 1.0

    return w, correction


# =====================================================================
# 修改 1: fft_no_stack (用于功能 2 的单周期 FFT)
# =====================================================================
def fft_no_stack(timeseries, sample_rate, window_type='hann', window_params=None):
    n_sam = timeseries.shape[-1]

    # 【核心临界点 1】：去直流偏置！把波形强行拉回 0 轴线
    timeseries_clean = timeseries - np.mean(timeseries, axis=-1, keepdims=True)

    w, corr = get_window_and_correction(n_sam, window_type, window_params)

    # 用去完直流的纯净交流波形来加窗
    windowed_ts = timeseries_clean * w
    yf_raw = np.fft.fft(windowed_ts, axis=-1)
    xf = np.fft.fftfreq(n_sam, 1.0 / sample_rate)

    yf = yf_raw * (2.0 / n_sam) * corr
    return xf, yf


# =====================================================================
# 2. 时域叠加 (支持将全长数据等分成多段，分别叠加)
# =====================================================================
def time_domain_stacking_segments(time_series, cyc_len, cyc_num, num_segments=1):
    chunk_cyc = cyc_num // num_segments
    stacked_waves = []

    for i in range(num_segments):
        start_idx = i * chunk_cyc * cyc_len
        end_idx = (i + 1) * chunk_cyc * cyc_len
        valid_data = time_series[start_idx:end_idx]
        matrix = valid_data.reshape((chunk_cyc, cyc_len))

        # 沿着列方向求平均，得到单周期波形
        stacked_wave = np.mean(matrix, axis=0)

        # 【新增修复】：在送去画图之前，先把这个单周期波形的直流偏置抽干！
        stacked_wave_clean = stacked_wave - np.mean(stacked_wave)

        stacked_waves.append(stacked_wave_clean)

    return stacked_waves, chunk_cyc


# =====================================================================
# 短序列 FFT (功能 2 用)
# =====================================================================
def fft_short(timeseries, sample_rate, window_type='rect', window_params=None):
    """
    对时域叠加后的单周期短序列进行 FFT。
    参数链已彻底打通，支持所有的自定义窗函数与参数。
    """
    # 【关键修复】：接收 window_params，并把它继续向下传给 fft_no_stack
    xf, yf = fft_no_stack(timeseries, sample_rate, window_type, window_params)
    return xf, np.abs(yf)


# =====================================================================
# 修改 2: fft_freq_stacking_segments (功能 3 的频域叠加)
# =====================================================================
def fft_freq_stacking_segments(time_series, sample_rate, cyc_len, cyc_num, window_type='rect', window_params=None,
                               num_segments=1):
    chunk_cyc = cyc_num // num_segments
    yf_stacked_list = []

    xf = np.fft.fftfreq(cyc_len, 1.0 / sample_rate)
    w, corr = get_window_and_correction(cyc_len, window_type, window_params)

    for i in range(num_segments):
        start_idx = i * chunk_cyc * cyc_len
        end_idx = (i + 1) * chunk_cyc * cyc_len
        valid_data = time_series[start_idx:end_idx]
        matrix = valid_data.reshape((chunk_cyc, cyc_len))

        # 【核心临界点 1】：矩阵每一行独立去直流
        matrix_clean = matrix - np.mean(matrix, axis=-1, keepdims=True)

        windowed_ts = matrix_clean * w
        yf_all_cycles = np.fft.fft(windowed_ts, axis=-1)
        yf_stacked_raw = np.mean(yf_all_cycles, axis=0)
        yf_stacked_list.append(yf_stacked_raw * (2.0 / cyc_len) * corr)

    return xf, yf_stacked_list, chunk_cyc


# =====================================================================
# 4. 加窗全长 FFT
# =====================================================================
def long_fft_with_window(time_series, sample_rate, window_type='hann', window_params=None):
    # 【关键修复】函数的括号里要加上 window_params=None
    # 并在调用 fft_no_stack 时继续把它传递下去
    freqs, yf_complex = fft_no_stack(time_series, sample_rate, window_type, window_params)
    return freqs, np.abs(yf_complex)