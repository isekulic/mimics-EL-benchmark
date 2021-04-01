import argparse
import pandas as pd
import json
import os
import torch

from IPython import embed
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertTokenizer
from sklearn.model_selection import train_test_split

class MimicsDataset(Dataset):
    def __init__(self, tokenizer, args, mode='dev'):
        self.tokenizer = tokenizer
        self.data_dir = args.data_dir
        self.mode = mode
        self.max_seq_len = args.max_seq_len
        self.text_input = args.text_input

        if args.click_explore == 'Click':
            df = pd.read_csv(os.path.join(self.data_dir, 'Click_titles_and_snippets_all.tsv'), sep='\t')
        elif args.click_explore == 'Explore':
            df = pd.read_csv(os.path.join(self.data_dir, 'Explore_titles_and_snippets.tsv'), sep='\t')
        df['question'] = df.question.replace('\"{2,}', '', regex=True)
        df = df[df['titles'].notna()]
        df = df[df['snippets'].notna()]

        if args.with_el_only:
            df = df[df['engagement_level'] > 0]

        qs = list(set(df['query'].unique()))
        X_train, X_dev = train_test_split(df, test_size=0.2, random_state=42)
        # X_train, X_dev = train_test_split(qs, test_size=0.2, random_state=42)

        if mode == 'train':
            self.X = X_train
            # self.X = df[df['query'].isin(X_train)]
        elif mode == 'dev':
            self.X = X_dev
            # self.X = df[df['query'].isin(X_dev)]
        elif mode == 'test':
            self.X = df


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tensors = self.example_to_tensor(idx)
        return tensors

    def example_to_tensor(self, idx):

        x = self.X.iloc[idx]
        query = x.query
        question = x.question
        answs = x[['option_1', 'option_2', 'option_3', 'option_4', 'option_5']].fillna('').str.cat(sep=' ')
        if 't' in self.text_input:
            second = x.titles
        elif 's' in self.text_input:
            second = x.snippets
        else:
            second = ''

        label = x.engagement_level / 10

        if 'qqa' in self.text_input:
            first = ' [SEP] '.join([query, question, answs])
        else:
            first = query

        encoded = self.tokenizer.encode_plus(first, second,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    truncation='only_second',
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_token_type_ids=True,
                    padding='max_length'
                        )

        encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])
        encoded['input_ids'] = torch.tensor(encoded['input_ids'])
        encoded['token_type_ids'] = torch.tensor(encoded['token_type_ids'])

        encoded.update({'label': torch.FloatTensor([label]),
                        'idx': torch.tensor(idx)})

        return encoded

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument('--data_dir', type=str, default='../data/')
        parser.add_argument('--mode', type=str, default='dev')
        parser.add_argument('--max_seq_len', type=int, default=512)
        return parser

class MimicsDatasetNrez(Dataset):
    def __init__(self, tokenizer, args, mode='dev'):
        self.tokenizer = tokenizer
        self.data_dir = args.data_dir
        self.mode = mode
        self.max_seq_len = args.max_seq_len
        self.text_input = args.text_input
        self.n_serp_elems = args.n_serp_elems # consider only top N SERP elements

        df = pd.read_csv(os.path.join(self.data_dir, 'Click_titles_and_snippets_all_sliced.tsv'), sep='\t')
        df['question'] = df.question.replace('\"{2,}', '', regex=True)
        df = df[df['titles'].notna()]
        df = df[df['snippets'].notna()]

        if args.with_el_only:
            df = df[df['engagement_level'] > 0]

        qs = list(set(df['query'].unique()))
        X_train, X_dev = train_test_split(df, test_size=0.2, random_state=42)
        # X_train, X_dev = train_test_split(qs, test_size=0.2, random_state=42)

        if mode == 'train':
            self.X = X_train
            # self.X = df[df['query'].isin(X_train)]
        elif mode == 'dev':
            self.X = X_dev
            # self.X = df[df['query'].isin(X_dev)]
        elif mode == 'test':
            self.X = df


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tensors = self.example_to_tensor(idx)
        return tensors

    def example_to_tensor(self, idx):

        x = self.X.iloc[idx]
        query = x.query
        question = x.question
        answs = x[['option_1', 'option_2', 'option_3', 'option_4', 'option_5']].fillna('').str.cat(sep=' ')
        if 't' in self.text_input:
            splt = x.titles.split('|#$')
        elif 's' in self.text_input:
            splt = x.snippets.split('|#$')

        second = ' '.join(splt[:self.n_serp_elems])

        label = x.engagement_level / 10

        if 'qqa' in self.text_input:
            first = ' [SEP] '.join([query, question, answs])
        else:
            first = query

        encoded = self.tokenizer.encode_plus(first, second,
                    add_special_tokens=True,
                    max_length=self.max_seq_len,
                    truncation='only_second',
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_token_type_ids=True,
                    padding='max_length'
                        )

        encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])
        encoded['input_ids'] = torch.tensor(encoded['input_ids'])
        encoded['token_type_ids'] = torch.tensor(encoded['token_type_ids'])

        encoded.update({'label': torch.FloatTensor([label]),
                        'idx': torch.tensor(idx)})

        return encoded

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument('--data_dir', type=str, default='../data/')
        parser.add_argument('--mode', type=str, default='dev')
        parser.add_argument('--max_seq_len', type=int, default=512)
        parser.add_argument('--n_serp_elems', type=int, default=10)
        return parser

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="MIMICS dataset")
    parser = MimicsDataset.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    cd = MimicsDataset(tokenizer, args, args.mode)
    embed()

