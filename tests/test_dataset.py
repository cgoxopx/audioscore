import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# print(sys.path)
import torch
import audioscore.dataset

ds = audioscore.dataset.AudioDataset_pkl_base("data/processed", "data/train_ds_4_al/denoise/0/test_score.json")
data = ds.__getitem__(0)[0]
print(type(data))
# for v in data:
#     print(v["延迟"])
#     print(v["原唱音频"].shape)
#     print(v["用户音频"].shape)
for k, v in data[0].items():
    print(k, type(v))

print(data[0]["用户前九泛音音量线"])
print(data[0]["原唱前九泛音音量线"].shape)
print(data[0]["原唱音高线"])
print(data[0]["用户音高线"].shape)
print(data[0]["原唱音量线"].shape)
print(data[0]["用户音量线"])
