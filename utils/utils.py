import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_embeddings(encoder, dataset, bs=128):
    embeddings = None
    dl = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8, drop_last=False)  # , pin_memory=True

    encoder = encoder.cuda().eval()

    with torch.no_grad():
        for i, (batch, _) in tqdm(enumerate(dl)):
            encoder = encoder.cuda()
            batch_embs = encoder(batch.cuda())

            if embeddings is None:
                embeddings = torch.zeros((len(dataset), batch_embs.shape[-1]))
            embeddings[i * bs : i * bs + batch_embs.shape[0]] = batch_embs

    dataset.set_embeddings(embeddings.float())
