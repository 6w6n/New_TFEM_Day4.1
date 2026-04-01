import sys
import numpy as np
from pathlib import Path

# 导入我们的核心功能模块
import data_io
import signal_processing
import visualization as vis


def main():
    # ==========================================
    # 1. 全局配置区
    # ==========================================
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST511.dat"

    print("\n🚀 欢迎使用 TFEM 时频电磁数据处理平台 🚀")
    print(f"[*] 默认发射机文件: {tx_filepath}")
    print(f"[*] 默认接收机文件: {rx_filepath}")

    # ==========================================
    # 2. 交互式控制主循环
    # ==========================================
    while True:
        print("\n" + "=" * 38)
        print("           📊 主 要 功 能 菜 单           ")
        print("=" * 38)
        print("  1. 时频数据导出 (长序列直接FFT)")
        print("  2. 叠加傅里叶变换-时域 (全部时域->叠加时域->FFT频谱)")
        print("  3. 叠加傅里叶变换-频域 (全部时域->全长FFT总体->频域叠加分析)")
        print("  4. [交互] 查看时域波形")
        print("  5. [交互] 查看全频段频谱 (自然数坐标)")
        print("  6. [测试] 不叠加，直接加窗长序列 FFT")
        print("  0. 退出系统")
        print("=" * 38)

        choice = input("👉 请输入操作对应的数字 [0-6]: ").strip()

        # ---------------------------------------------------------
        if choice == '1':
            print("\n>>> 正在执行: [1] 时频数据导出并进行单周期FFT...")
            data_io.process_and_export_all_data(tx_filepath, rx_filepath)

        # ---------------------------------------------------------
        elif choice == '2':
            print("\n>>> 正在执行: [2] 叠加傅里叶变换 - 时域...")
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue

            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue

            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])
            theory_f0 = rx_sr / cyc_len

            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print("\n👉 请选择数据切分叠加的方式: [1]不切分(1段)  [2]切分为2段  [4]切分为4段  [10]切分为10段")
            seg_choice = input("请输入段数编号 [默认1]: ").strip() or '1'
            num_segments = int(seg_choice) if seg_choice in ['1', '2', '4', '10'] else 1

            # --- 核心：十大窗函数实验菜单 ---
            print("\n👉 请选择短序列的加窗策略 (⚠️ 实验对比推荐，真理越辩越明):")
            print("  [1] Rect (矩形窗): 理论上的唯一正确解")
            print("  [2] Hann (汉宁窗)     [3] Hamming (汉明窗)  [4] Blackman")
            print("  [5] Gaussian         [6] Kaiser           [7] Flattop")
            print("  [8] Tukey (图基窗)    [9] Nuttall          [10] Chebyshev")

            win_choice = input("请输入编号 [默认1]: ").strip() or '1'
            win_map = {
                '1': 'rect', '2': 'hann', '3': 'hamming', '4': 'blackman',
                '5': 'gaussian', '6': 'kaiser', '7': 'flattop',
                '8': 'tukey', '9': 'nuttall', '10': 'chebwin'
            }
            win_type = win_map.get(win_choice, 'rect')
            win_params = None

            if win_type == 'gaussian':
                win_params = float(input("  请输入 Gaussian 窗的标准差 (默认 10.0): ").strip() or 10.0)
            elif win_type == 'kaiser':
                win_params = float(input("  请输入 Kaiser 窗的 Beta 参数 (默认 14.0): ").strip() or 14.0)
            elif win_type == 'tukey':
                win_params = float(input("  请输入 Tukey 窗的 Alpha 比例 (默认 0.25): ").strip() or 0.25)
            elif win_type == 'chebwin':
                win_params = float(input("  请输入 Chebyshev 窗压制级别 (dB, 默认 100.0): ").strip() or 100.0)

            stacked_waves_list, chunk_num = signal_processing.time_domain_stacking_segments(
                time_series, cyc_len, cyc_num, num_segments=num_segments
            )

            ch_name = "Ex" if ch_idx == 0 else "Hz"
            labels = [f"段 {i + 1} (包含 {chunk_num} 个循环)" for i in range(num_segments)]

            freqs = None
            spectrum_list = []
            for wave in stacked_waves_list:
                # 把你选的窗函数传到底层！
                f, s = signal_processing.fft_short(wave, rx_sr, window_type=win_type, window_params=win_params)
                if freqs is None: freqs = f
                spectrum_list.append(s)

            print(f"[√] 已完成 {num_segments} 段叠加计算 (Win: {win_type})！正在出图...")

            vis.plot_multi_waveform(
                stacked_waves_list,
                title=f"Time-Domain Multi-Segment Stacking Comparison - Ch {ch_idx + 1} ({ch_name})",
                labels=labels,
                ylim=[-1000, 5000]
            )

            vis.plot_multi_analyzed_spectrum(
                freqs_fft=freqs, yf_list=spectrum_list, fundamental_freq=f0, num_harmonics=15,
                title=f"Stacked Spectrum (Time-Domain | Win: {win_type.capitalize()}) - Ch {ch_idx + 1} ({ch_name})",
                labels=labels
            )

        # ---------------------------------------------------------
        elif choice == '3':
            print("\n>>> 正在执行: [3] 叠加傅里叶变换 - 频域...")
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue

            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue

            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])
            theory_f0 = rx_sr / cyc_len

            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print("\n👉 请选择数据切分叠加的方式: [1]不切分(1段)  [2]切分为2段  [4]切分为4段  [10]切分为10段")
            seg_choice = input("请输入段数编号 [默认1]: ").strip() or '1'
            num_segments = int(seg_choice) if seg_choice in ['1', '2', '4', '10'] else 1

            # --- 核心：十大窗函数实验菜单 ---
            print("\n👉 请选择短序列单周期 FFT 窗函数 (⚠️ 实验对比推荐):")
            print("  [1] Rect (矩形窗): 理论上的唯一正确解")
            print("  [2] Hann (汉宁窗)     [3] Hamming (汉明窗)  [4] Blackman")
            print("  [5] Gaussian         [6] Kaiser           [7] Flattop")
            print("  [8] Tukey (图基窗)    [9] Nuttall          [10] Chebyshev")

            win_choice = input("请输入编号 [默认1]: ").strip() or '1'
            win_map = {
                '1': 'rect', '2': 'hann', '3': 'hamming', '4': 'blackman',
                '5': 'gaussian', '6': 'kaiser', '7': 'flattop',
                '8': 'tukey', '9': 'nuttall', '10': 'chebwin'
            }
            win_type = win_map.get(win_choice, 'rect')
            win_params = None

            if win_type == 'gaussian':
                win_params = float(input("  请输入 Gaussian 窗的标准差 (默认 10.0): ").strip() or 10.0)
            elif win_type == 'kaiser':
                win_params = float(input("  请输入 Kaiser 窗的 Beta 参数 (默认 14.0): ").strip() or 14.0)
            elif win_type == 'tukey':
                win_params = float(input("  请输入 Tukey 窗的 Alpha 比例 (默认 0.25): ").strip() or 0.25)
            elif win_type == 'chebwin':
                win_params = float(input("  请输入 Chebyshev 窗压制级别 (dB, 默认 100.0): ").strip() or 100.0)

            # 把你选的窗函数传到底层！
            freqs_stacked, spectrum_list, chunk_num = signal_processing.fft_freq_stacking_segments(
                time_series, rx_sr, cyc_len, cyc_num, window_type=win_type, window_params=win_params,
                num_segments=num_segments
            )

            ch_name = "Ex" if ch_idx == 0 else "Hz"
            labels = [f"段 {i + 1} (包含 {chunk_num} 个循环)" for i in range(num_segments)]

            print(f"[√] 已完成 {num_segments} 段频域均值叠加 (Win: {win_type})！正在出图...")

            vis.plot_multi_analyzed_spectrum(
                freqs_fft=freqs_stacked, yf_list=spectrum_list, fundamental_freq=f0, num_harmonics=15,
                title=f"Stacked Spectrum (Freq-Domain | Win: {win_type.capitalize()}) - Ch {ch_idx + 1} ({ch_name})",
                labels=labels
            )
        # ---------------------------------------------------------
        elif choice == '4':
            print("\n>>> 开启交互式时域查看器...")
            inp = input("👉 要查看哪一端的时域数据 [T/R]: ").strip().upper()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                ts_dir = Path(tx_filepath).parent / f"{stem}_Timeseries"
                vis.interactive_time_viewer(ts_dir, stem)
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"
                vis.interactive_time_viewer(ts_dir, stem)
            else:
                print("❌ 输入无效！")

        # ---------------------------------------------------------
        elif choice == '5':
            print("\n>>> 开启交互式频域查看器...")
            inp = input("👉 要查看哪一端的频谱数据 [T/R]: ").strip().upper()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                fs_dir = Path(tx_filepath).parent / f"{stem}_FreqSeries"
                vis.interactive_freq_viewer(fs_dir, stem)
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                fs_dir = Path(rx_filepath).parent / f"{stem}_FreqSeries"
                vis.interactive_freq_viewer(fs_dir, stem)
            else:
                print("❌ 输入无效！")

                # ---------------------------------------------------------
        elif choice == '6':
                print("\n>>> 正在执行: [6] 测试 - 不叠加，直接加窗长序列 FFT...")
                stem = Path(rx_filepath).stem
                ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

                try:
                    p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                    ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
                except ValueError:
                    print("❌ 输入无效，请输入整数！")
                    continue

                target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
                if not target_file.exists():
                    print(f"❌ 找不到文件: {target_file}")
                    continue

                data = np.loadtxt(target_file, skiprows=2)
                time_series = data[:, ch_idx]

                # 去直流偏置，防止全长 FFT 零频泄漏
                time_series = time_series - np.mean(time_series)

                rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
                cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
                theory_f0 = rx_sr / cyc_len

                user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
                f0 = theory_f0 if not user_f0 else float(user_f0)

                # --- 新增：选择 6 大窗函数与参数配置 ---
                print("\n👉 请选择长序列的加窗策略 (抗频域泄漏与工频干扰):")
                print("  [1] Rect (矩形窗): 分辨率最高，但不抑制泄漏")
                print("  [2] Hann (汉宁窗): 经典平滑窗，旁瓣衰减快")
                print("  [3] Hamming (汉明窗): 类似汉宁，第一旁瓣压制更深")
                print("  [4] Blackman (布莱克曼窗): 极强的旁瓣抑制，主瓣变宽")
                print("  [5] Gaussian (高斯窗): 可调标准差，时频局部化极佳")
                print("  [6] Kaiser (凯赛窗): 工业级可调 Beta，极致控制泄漏")
                win_choice = input("请输入编号 [默认2]: ").strip() or '2'

                # 根据用户选择映射窗函数名称
                win_map = {'1': 'rect', '2': 'hann', '3': 'hamming',
                           '4': 'blackman', '5': 'gaussian', '6': 'kaiser'}
                win_type = win_map.get(win_choice, 'hann')
                win_params = None

                # 处理需要额外参数的高级窗函数
                if win_type == 'gaussian':
                    g_val = input("  请输入 Gaussian 窗的标准差 (推荐 50~200, 默认 100): ").strip()
                    win_params = float(g_val) if g_val else 100.0
                elif win_type == 'kaiser':
                    b_val = input("  请输入 Kaiser 窗的 Beta 参数 (推荐 8~14, 默认 14.0): ").strip()
                    win_params = float(b_val) if b_val else 14.0

                print(f"  [>] 原始序列总点数: {len(time_series)} 点")
                param_str = f", 参数={win_params}" if win_params else ""
                print(f"  [>] 正在应用 {win_type.capitalize()} 窗{param_str} 并执行全长 FFT...")

                # 调用底层引擎
                freqs, amplitude = signal_processing.long_fft_with_window(
                    time_series, rx_sr, window_type=win_type, window_params=win_params
                )

                print("[√] 计算完成！正在出图...")

                # 自动分配通道名称并生成标题
                ch_name = "Ex" if ch_idx == 0 else "Hz"
                title_text = f"Unstacked Long-Sequence FFT (Win: {win_type.capitalize()}) - Ch {ch_idx + 1} ({ch_name})"

                vis.plot_analyzed_spectrum(
                    freqs_fft=freqs,
                    yf_fft=amplitude,
                    fundamental_freq=f0,
                    num_harmonics=15,
                    title=title_text
                )

        # ---------------------------------------------------------
        elif choice == '0':
            print("\n👋 感谢使用，系统已退出！")
            sys.exit(0)

        else:
            print("\n❌ 错误: 无效的输入，请输入 0 到 6 之间的数字！")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 强制中断，系统已退出！")
        sys.exit(0)