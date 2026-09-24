import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

emotions = ['happy','angry','sad','disgust','fear','neutral', 'surprise']

class FER2013Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        data = []
        labs = []
        for idx, emotion in enumerate(emotions):
            dir = Path(path+'/fer2013/'+split+'/'+emotion)
            for item in dir.iterdir():
                data.append(plt.imread(item))
                labs.append(idx)

        self.x = torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(np.array(labs), dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


