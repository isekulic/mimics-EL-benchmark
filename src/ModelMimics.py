from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer, AlbertModel, AlbertForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from sklearn import metrics

import math
import pandas as pd
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from MimicsDataset import MimicsDataset, MimicsDatasetNrez

from IPython import embed

class ModelMimics(pl.LightningModule):
    def __init__(self, hparams):
        super(ModelMimics, self).__init__()
        self.hparams = hparams

        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.dropout = nn.Dropout(hparams.classifier_dropout_prob)
        self.classifier = nn.Linear(self.albert.config.hidden_size, 1)

        print('Loaded model')

    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False):

        out = self.albert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_attentions=output_attentions)
        if output_attentions: 
            last_hidden, pooled_output, attentions = out
        else:
            last_hidden, pooled_output = out
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_attentions: 
            return logits, attentions[-1]
        else:
            return logits

    def configure_optimizers(self):
        # not_albert_params = [p for name, p in filter(
                    # lambda t: not t[0].startswith('albert'),
                    # self.named_parameters())]
        optimizer = AdamW([
                    {'params': self.albert.parameters(),
                    'lr': 1e-6},
                    {'params': self.classifier.parameters()}],
                    lr=self.hparams.lr,
                    betas=(0.9, 0.999), weight_decay=0.01)
        # scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                            # self.hparams.num_warmup_steps,
                            # self.hparams.num_training_steps),
                    # 'interval': 'step',
                    # 'name': 'linear_with_warmup'}
        return optimizer
        # return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = MimicsDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='train'
                        )

        sampler = None
        self.train_dataloader_object = DataLoader(
                                    dataset, batch_size=self.hparams.data_loader_bs,
                                    shuffle=(sampler is None),
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    collate_fn=ModelMimics.collate_fn
                                    )
        return self.train_dataloader_object

    def val_dataloader(self):
        dataset = MimicsDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='dev'
                            )

        sampler = None
        self.val_dataloader_object = DataLoader(
                                    dataset, batch_size=self.hparams.val_data_loader_bs,
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    collate_fn=ModelMimics.collate_fn
                                    )
        return self.val_dataloader_object

    def test_dataloader(self):
        dataset = MimicsDataset(
                        tokenizer=self.tokenizer,
                        args=self.hparams,
                        mode='dev'
                            )

        sampler = None
        self.test_dataloader_object = DataLoader(
                                    dataset, batch_size=self.hparams.val_data_loader_bs,
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers, sampler=sampler,
                                    collate_fn=ModelMimics.collate_fn
                                    )
        return self.test_dataloader_object

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.mse_loss(output, labels)
        if self.logger:
            self.logger.log_metrics({'train_loss': loss.item()})

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output = self.forward(input_ids, attention_mask, token_type_ids)

        loss = F.mse_loss(output, labels)
        if self.logger:
            self.logger.log_metrics({'val_loss': loss.item()})
        return {'loss': loss, 'pred': output, 'idxs': idxs, 'labels': labels}

    def validation_epoch_end(self, outputs):
        """ 
        outputs: dict of outputs of validation_step (or validation_step_end in dp/ddp2)
        outputs['loss'] --> losses of all the batches
        outputs['pred'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        pred = torch.cat([x['pred'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        loss = F.mse_loss(pred, labels)

        # if self.logger:
            # self.logger.log_metrics(scores)
        scores = self.evaluate(labels, pred)
        r2 = torch.Tensor([scores['r2']])

        print(f"\nDEV:: avg-LOSS: {avg_loss} || {loss}", scores)

        # return {'val_epoch_loss': avg_loss, 'scores':scores, 'r2': torch.Tensor([scores['r2']])}
        return {'val_epoch_loss': avg_loss, 'scores':scores, 'r2': r2}

    def evaluate(self, y, y_pred):
        y = y.squeeze().cpu().numpy()
        y_pred = y_pred.squeeze().cpu().numpy()
        ret_d = {}
        for name, metric in zip(['mae', 'mse', 'r2'], [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.r2_score]):
            val = metric(y, y_pred)
            ret_d[name] = val
        return ret_d

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([x['input_ids'] for x in batch])
        token_type_ids = torch.stack([x['token_type_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])

        label = torch.stack([x['label'] for x in batch])
        idx = torch.stack([x['idx'] for x in batch])

        return (input_ids, attention_mask, token_type_ids, label, idx)



    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels, idxs = batch
        output, attentions = self.forward(input_ids, attention_mask, token_type_ids, output_attentions=True)
        # embed()

        return {'pred': output, 'idxs': idxs, 'labels': labels}
        # return {'pred': output, 'idxs': idxs, 'labels': labels, 'attentions': attentions.cpu()}
    
    def test_epoch_end(self, outputs):
        """ 
        outputs: dict of outputs of test_step (or test_step_end in dp/ddp2)
        outputs['pred'] --> scores for each example
        outputs['idxs'] --> indexes in Dataset to connect with scores
        """

        pred = torch.cat([x['pred'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        # if self.hparams.save_attentions:
        # attentions = torch.cat([x['attentions'] for i, x in enumerate(outputs) if i < 3]).cpu()
        # idxs = torch.cat([x['idxs'] for i, x in enumerate(outputs) if i < 3]).cpu()
        # pickle.dump(attentions, open(f'/scratch/sekulic/mimics/attentions-{self.hparams.text_input}.pickle', 'wb'))
        # pickle.dump(idxs, open(f'/scratch/sekulic/mimics/idxs-{self.hparams.text_input}.pickle', 'wb'))

        scores = self.evaluate(labels, pred)

        idxs = torch.cat([x['idxs'] for x in outputs]).cpu().tolist()
        pred = pred.cpu().tolist()
        
        df = self.test_dataloader_object.dataset.X.iloc[idxs]
        df['pred'] = sum(pred, [])

        df.to_csv(f'/scratch/sekulic/mimics/results-final-{self.hparams.text_input}-{self.hparams.n_serp_elems}.tsv', sep='\t', header=False, index=False)
        return {}

