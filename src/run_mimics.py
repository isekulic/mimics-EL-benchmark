import argparse
import os
import pytorch_lightning as pl
import torch

from ModelMimics import ModelMimics
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from IPython import embed

seed = 42
seed_everything(seed)

def main(hparams):
    model = ModelMimics(hparams)
    if hparams.mode == 'test':

        checkpoint = torch.load(hparams.test_ckpt)
        model.load_state_dict(checkpoint['state_dict'])

    loggers = []
    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger("tb_logs", name=f"{hparams.run_name}",
                                    version=hparams.slurm_job_id)
        loggers.append(tb_logger)

    checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), 'checkpoints'),
                save_top_k=1,
                verbose=True,
                monitor='r2',
                mode='max',
                prefix=''
                )

    trainer = pl.Trainer(
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            # distributed_backend=hparams.distributed_backend,
            # control the effective batch size with this param
            accumulate_grad_batches=hparams.trainer_batch_size,
            # Training will stop if max_steps or max_epochs have reached (earliest).
            max_epochs=hparams.epochs,
            max_steps=hparams.num_training_steps, 
            logger=loggers,
            checkpoint_callback=checkpoint_callback,
            # progress_bar_callback=False,
            # progress_bar_refresh_rate=0,
            # use_amp=True --> use 16bit precision
            # val_check_interval=20, # val 4 times during 1 train epoch
            # val_check_interval=hparams.val_check_interval, # val every N steps
            # num_sanity_val_steps=5,
            # fast_dev_run=True
        )

    if hparams.mode == 'test':
        trainer.test(model)
    elif hparams.mode == 'train':
        trainer.fit(model)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='MIMICS')
    # MODEL SPECIFIC
    parser.add_argument("--model_name", type=str, default='albert-base-v2',
                        help="Full name of: bert|albert|longformer")
    # parser.add_argument('--ckp_path', type=str, default='albert-marco.ckp',
                        # help='checkpoint to load the data from')
    parser.add_argument("--hidden_size", type=int, default=768,
                        help='Hidden size of transformer, and input to pre_clf.')
    parser.add_argument("--classifier_dropout_prob", type=float, default=0.1,
                        help='Classifier dropout probability.')
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum number of wordpieces of the sequence")
    # parser.add_argument("--num_warmup_steps", type=int, default=50)
    parser.add_argument("--num_training_steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--val_check_interval", type=int, default=0.5,
                        help='Run through dev set every N steps, or portion of tr epoch.')
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Num subprocesses for DataLoader")

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_ckpt', type=str, default='_ckpt_epoch_0_v6.ckpt')
    parser.add_argument('--run_name', type=str, default='mimics',
                        help='run name')

    # EXPERIMENT SPECIFIC
    parser.add_argument("--data_dir", type=str, default='../data/',)
    # effective batch size will be: 
    # trainer_batch_size * data_loader_bs
    parser.add_argument("--trainer_batch_size", type=int, default=6,
                        help='Batch size for Trainer. Accumulates grads every k batches')
    parser.add_argument("--data_loader_bs", type=int, default=6,
                        help='Batch size for DataLoader object')
    parser.add_argument("--val_data_loader_bs", type=int, default=18,
                        help='Batch size for validation data loader. If not specified,\
                        --data_loader_bs is used.')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--slurm_job_id", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0, 
                        help='Use Weights&Biases (wandb) logger')
    parser.add_argument("--use_tensorboard", type=int, default=1,
                        help='Use TensorBoard logger (default in PL)')

    # Distributed training
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Num nodes allocated by SLURM")
    parser.add_argument("--distributed_backend", type=str, default='ddp',
                        help="Use distributed backend: dp/ddp/ddp2")
    parser.add_argument("--text_input", type=str, default='qqat',
                        help="qqast -- query, question, answers, snippets, titles. Chose a combination.")
    parser.add_argument("--save_attentions", type=int, default=0,
                        help="save attentions after val epoch end")
    parser.add_argument("--click_explore", type=str, default='Click',
                        help="Click | Explore")
    parser.add_argument("--with_el_only", type=int, default=0,
                        help="Subset of dataset with EL > 0")
    parser.add_argument('--n_serp_elems', type=int, default=10)

    hparams = parser.parse_args()
    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs
    print(hparams)
    main(hparams)

