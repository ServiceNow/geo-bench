import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_embeddings(encoder, dataset):
    embeddings = None
    dl = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, drop_last=False)  # , pin_memory=True

    encoder = encoder.cuda()

    with torch.no_grad():
        for i, (batch, _) in tqdm(enumerate(dl)):

            batch_embs = encoder(batch.cuda())

            if embeddings is None:
                embeddings = torch.zeros((len(dataset), batch_embs.shape[-1]))
            embeddings[i * 128 : i * 128 + batch_embs.shape[0]] = batch_embs

    dataset.set_embeddings(embeddings.float())
