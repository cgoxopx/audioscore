import os

import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import librosa
import csv
import torch

def calculate_volume(frame):
    """计算音频帧的音量（RMS）"""
    return np.sqrt(np.mean(np.square(np.abs(frame))))

def process_audio(input_path, output_dir, threshold=0.01, frame_size=1024, segment_duration=10):
    """
    处理音频文件，检测高音量片段并截取
    
    参数:
        input_path: 输入音频路径
        output_dir: 输出目录
        threshold: 音量阈值 (0-1范围)
        frame_size: 检测帧大小（采样点数）
        segment_duration: 截取片段时长（秒）
    """
    # 读取音频文件
    data, sr = sf.read(input_path, dtype='float32')
    num_channels = data.shape[1] if len(data.shape) > 1 else 1

    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    data = pyln.normalize.loudness(data, loudness, -12.0)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    current_pos = 0
    segment_count = 0
    total_samples = len(data)
    segment_samples = int(segment_duration * sr)
    
    print(f"处理�***REMOVED***始: {input_path}")
    print(f"采样率: {sr}Hz, 总时长: {total_samples/sr:.1f}秒")
    print(f"使用阈值: {threshold:.3f}, 帧大小: {frame_size}采样点")

    out_files = []

    while current_pos + frame_size <= total_samples:
        # 提取当前帧
        start = current_pos
        end = start + frame_size
        frame = data[start:end]
        
        # 计算音量（多声道取平均）
        if num_channels > 1:
            channel_volumes = [calculate_volume(frame[:, c]) for c in range(num_channels)]
            volume = np.mean(channel_volumes)
        else:
            volume = calculate_volume(frame)

        # 检测阈值
        if volume > threshold:
            # 计算截取结束位置
            segment_end = min(start + segment_samples, total_samples)
            
            # 写入片段文件
            output_path = os.path.join(
                output_dir,
                f"segment_{segment_count:03d}_start{start/sr:.1f}s.wav"
            )
            sf.write(output_path, data[start:segment_end], sr)

            out_files.append(os.path.abspath(output_path))
            
            print(f"检测到高音量片段: {output_path} (时长: {segment_end/sr - start/sr:.1f}秒)")
            
            # 更新指针到片段末尾
            current_pos = segment_end
            segment_count += 1
        else:
            # 移动到下一帧
            current_pos += frame_size

    print(f"处理完成，共找到{segment_count}个有效片段")
    return out_files

def random_cut(input_path, sr, segment_duration=30):
    # 从音频中随机切出30秒
    # 读取音频文件
    data, sr = librosa.load(input_path, sr=sr, mono=True)

    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    data = pyln.normalize.loudness(data, loudness, -12.0)

    total_samples = len(data)
    segment_samples = int(segment_duration * sr)

    print(input_path, total_samples - segment_samples, total_samples , segment_samples)

    start = np.random.randint(0, total_samples - segment_samples)
    end = start + segment_samples

    return data[start:end], sr

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch

def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * \
        np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
        (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (
        f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min(
    ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse
