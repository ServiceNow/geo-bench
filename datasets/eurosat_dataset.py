from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


class EurosatDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.embeddings = None

        with open(self.root / f"{split}.txt") as f:
            filenames = f.read().splitlines()

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for fn in filenames:
            cls_name = fn.split("_")[0]
            self.samples.append(self.root / cls_name / fn)

    def __getitem__(self, index):

        path = self.samples[index]
        target = self.class_to_idx[path.parts[-2]]

        if self.embeddings is not None:
            return self.embeddings[index], target

        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
