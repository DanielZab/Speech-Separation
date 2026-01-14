from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
import glob


class CustomDataset(Dataset):
    def __init__(self, root_dir, sub_dirs):
        self.root_dir = root_dir
        self.sub_dirs = sub_dirs
        self.data_s1 = []
        self.data_s2 = []
        self.data_mix = []
        paths = ["mix_clean", "s1", "s2"]
        for sub_dir in sub_dirs:
            cur_path = os.path.join(root_dir, sub_dir)
            tmp = [glob.glob(os.path.join(cur_path, path, "*.wav")) for path in paths]
            tmp = [set(os.path.basename(x) for x in f) for f in tmp]
            for i in range(1, len(tmp)):
                assert tmp[0] == tmp[i], "File names do not match across directories"
            names = list(tmp[0])

            for path, data_list in zip(
                paths, [self.data_mix, self.data_s1, self.data_s2]
            ):
                tmp = map(lambda x: os.path.join(cur_path, path, x), names)
                data_list.extend(tmp)
        assert len(self.data_mix) == len(self.data_s1) == len(self.data_s2), (
            "Data length mismatch"
        )

    def __len__(self):
        return len(self.data_mix)

    def __getitem__(self, idx):
        return (
            torchaudio.load(self.data_mix[idx])[0],
            torchaudio.load(self.data_s1[idx])[0],
            torchaudio.load(self.data_s2[idx])[0],
        )
