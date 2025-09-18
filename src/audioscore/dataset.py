import os
import torch
import json
import pickle as pickle
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import numpy

torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.core.multiarray.scalar,
    numpy.dtype,
    numpy.ndarray,
    numpy.dtypes.Float32DType])

class AudioDataset_pkl_base(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_bags = pickle.load(open(file_path, "rb"))
        audio_id = file_path.split("/")[-1].split("对比数据")[0]
        if audio_id in self.target:
            score = self.target[audio_id] - 1 #id要减一
        else:
            score = 4
            for i in range(len(data_bags)):
                data_bags[i]["用户音频特征"] = data_bags[i]["原唱音频特征"]
                data_bags[i]["用户前九泛音音量线"] = data_bags[i]["原唱前九泛音音量线"]
                data_bags[i]["用户音高线"] = data_bags[i]["原唱音高线"]
                data_bags[i]["用户音量线"] = data_bags[i]["原唱音量线"]
        return data_bags, score, file_path
        
class AudioDataset_fullaudio(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        data_bags = pickle.load(open(file_path, "rb"))
        audio_id = file_path.split("/")[-1].split("对比数据")[0]
        if audio_id in self.target:
            score = self.target[audio_id] - 1 #id要减一
        else:
            score = 4
            data_bags["res_user"] = data_bags["res_ori"]
        return data_bags, score, file_path
        
class AudioDataset_tensor(Dataset):
    def __init__(self, dir_path, target_file, use_same_data=False):
        
        self.target = dict()
        with open(target_file, "r") as f:
            target = json.load(f)
            for item in target:
                audio_id = item["audio"].split("/")[-1].split(".")[0]
                self.target[audio_id] = item["score"]

        # walk dir_path
        self.files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        # data = pickle.load(f)
                        # self.data.extend(data)
                        audio_id = file.split("/")[-1].split("对比数据")[0]
                        if audio_id in self.target or use_same_data:
                            self.files.append(os.path.join(root, file))


        print(f"Found {len(self.files)} files in {dir_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data_bags = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
            audio_id = file_path.split("/")[-1].split("对比数据")[0]
            if audio_id in self.target:
                score = self.target[audio_id] - 1 #id要减一
            else:
                score = 4
                data_bags["res_user"] = data_bags["res_ori"]
                if "audio_ori" in data_bags:
                    data_bags["audio_user"] = data_bags["audio_ori"]
                if "wespeaker_ori" in data_bags:
                    data_bags["wespeaker_user"] = data_bags["wespeaker_ori"]
                if "samoye_ori" in data_bags:
                    data_bags["samoye_user"] = data_bags["samoye_ori"]
                if "spk_ori" in data_bags:
                    data_bags["spk_user"] = data_bags["spk_ori"]
            
            return data_bags, score, file_path
        except Exception as e:
            print(file_path)
            print(e)
            raise Exception(f"Error in loading {file_path}")