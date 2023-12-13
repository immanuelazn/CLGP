import os
#import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
#import albumentations as A
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
import triton
import logging
import os

from huggingface_hub import login
import transformers
login('ENTER LOGIN')

data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'datasets', 'refseq')
save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'output', 'clip')
class CFG:
    debug = False
    save_path = save_path
    train_data = os.path.join(data_path, "train_captions.csv")
    test_data = os.path.join(data_path, "test_captions.csv")
    sequence_path = os.path.join(data_path, "test.fa") # TODO
    captions_path = os.path.join(data_path, "sequence_captions_test_space.json")
    batch_size = 16
    num_workers = 4
    head_lr = 1e-3
    nucleotide_encoder_lr = 1e-5
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "immanuelabdi/DNABERT-2-117M"
    nucleotide_embedding = 768
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

### Utils
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



#Model
class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self. te = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        return last_hidden_state[:, self.target_token_idx, :]

class NucleotideEncoder(nn.Module):
    def __init__(self, model_name=CFG.model_name,
                 trainable=CFG.trainable):
        super().__init__()
        logging.info("Loading DNABERT2.  Note, this is using a custom version of DNABERT2 that is not available on HuggingFace."
                     "use the model_name 'immanuelabdi/DNABERT-2-117M'")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        return last_hidden_state[:, self.target_token_idx, :]


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sequences,
                 captions,
                 tokenizer_sequence,
                 tokenizer_captions):
        """
        sequence_filenames and cpations must have the same length.
        """
        self.sequences = sequences
        self.captions = captions
        self.encoded_sequences = tokenizer_sequence(
            list(sequences), padding=True, truncation=True, max_length=300
        )
        self.encoded_captions = tokenizer_captions(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )

    def __getitem__(self, idx):
        item_nucleotide = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_sequences.items()
        }
        item_caption = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        return {
            "input_ids_nucleotide": item_nucleotide["input_ids"],
            "attention_mask_nucleotide": item_nucleotide["attention_mask"],
            "input_ids_text": item_caption["input_ids"],
            "attention_mask_text": item_caption["attention_mask"],
        }

    def __len__(self):
        return len(self.captions)

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        nucleotide_embedding=CFG.nucleotide_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.nucleotide_encoder = NucleotideEncoder(CFG.model_name, CFG.trainable)
        self.text_encoder = TextEncoder()
        self.nucleotide_projection = ProjectionHead(embedding_dim=nucleotide_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting nucleotide and Text Features
        nucleotide_features = self.nucleotide_encoder(
            input_ids=batch["input_ids_nucleotide"], attention_mask=batch["attention_mask_nucleotide"]
        )
        text_features = self.text_encoder(
            input_ids=batch["input_ids_text"], attention_mask=batch["attention_mask_text"]
        )
        # Getting nucleotide and Text Embeddings (with same dimension)
        nucleotide_embeddings = self.nucleotide_projection(nucleotide_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ nucleotide_embeddings.T) / self.temperature
        nucleotide_similarity = nucleotide_embeddings @ nucleotide_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (nucleotide_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        nucleotide_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (nucleotide_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
def build_loaders(dataframe,
                  tokenizer_nucleotide,
                  tokenizer_caption,
                  mode):
    dataset = CLIPDataset(
        dataframe["sequence"].values,
        dataframe["caption"].values,
        tokenizer_nucleotide,
        tokenizer_caption,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


from sklearn.model_selection import train_test_split


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["input_ids_nucleotide"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["input_ids_nucleotide"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    int_df = pd.read_csv(CFG.train_data, dtype={"sequence": str, "caption": str})
    train_df, valid_df = train_test_split(int_df,
                                          test_size=0.03)
    logging.info('Building Tokenizers')
    tokenizer_captions = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokenizer_nucleotide = AutoTokenizer.from_pretrained(CFG.model_name)
    logging.info('Building DataLoaders')
    train_loader = build_loaders(train_df,
                                 tokenizer_nucleotide,
                                 tokenizer_captions, mode="train")
    valid_loader = build_loaders(valid_df,
                                 tokenizer_nucleotide,
                                 tokenizer_captions, mode="valid")
    logging.info('Building Model')
    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.nucleotide_encoder.parameters(), "lr": CFG.nucleotide_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.nucleotide_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    logging.info('Starting Training')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), save_path)
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    main()