import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio, concatenate_datasets
import numpy as np
from typing import Dict
import os


"""
You can select any audio dataset on huggingface, Here we would use multilingual librispeech
ensure the audio column is named 'audio'
You could also try multilingual audio, concatenating audios of different languages
multilingual librispeech is almost 100GB. 

"""
languages = [
    'dutch',
    'french',
    'german',
    'italian',
    'spanish',
]

class Data(Dataset):
    def __init__(self, data):
        self.data = data

        self.max_length  = 16000 * 10
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        audio = self.data[index]['audio'][:self.max_length]
        return {'audio': audio}


def collate_fn(batch):
    audio  = [torch.FloatTensor(i['audio']) for i in batch]
    audio  = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
    return {'inputs':audio}


def get_loaders(batch_size = 16):
    sets = [load_dataset('facebook/multilingual_librispeech', i,split='train', num_proc = os.cpu_count())
            for i in languages]
    dataset = concatenate_datasets(sets).shuffle()
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16_000))
    dataset = dataset.train_test_split(test_size = 0.01)
    train_dataset = Data(dataset['train'])
    test_dataset = Data(dataset['test'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers = os.cpu_count())
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers = os.cpu_count())
    return train_loader, valid_loader