# mimics-EL-benchmark
Code and runs for the ECIR 2021 paper "User Engagement Prediction for Clarification in Search".

## Download dataset

You can download the preprocessed MIMICS-Click with mapped SERP elements for each query from [here](https://usi365-my.sharepoint.com/:f:/g/personal/sekuli_usi_ch/El51JpycjZdEj1R8neF1dQ4BJXCSqAK5RhKb0BRfwrmt7Q?e=2gNqgY).

## Training

Run `run_mimics.py` with wanted parameters. Parameters are described in the `run_mimics.py` file. 

## Testing

Download the checkpoints used in the paper [here](https://usi365-my.sharepoint.com/:f:/g/personal/sekuli_usi_ch/El51JpycjZdEj1R8neF1dQ4BJXCSqAK5RhKb0BRfwrmt7Q?e=2gNqgY). You can then run the prediction with:

`python run_mimics.py --mode 'test' --test_ckpt '_ckpt_epoch_0_v6.ckpt' --text_input 'qqat'`

## Citation

Please cite the following if you found our paper or these resources useful.

```bibtex
@inproceedings{sekulic-2021-user,
  author    = {Ivan Sekuli{\'c}
                and Mohammad Aliannejadi
                and Fabio Crestani},
  title     = {User Engagement Prediction for Clarification in Search},
  booktitle = {Advances in Information Retrieval - 43rd European Conference on {IR} Research, 
    {ECIR} 2021, Virtual Event, March 28 - April 1, 2021, Proceedings},
  year      = {2021}
}
```
