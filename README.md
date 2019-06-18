# Combined loss for verification

Combining metric-learning and multi-clss classification for speaker dependent representation learning from speech

## Requirements

```
Python = 3.6
Pytorch >= 1.0.0
Scikit-learn >=0.19
tqdm
h5py
```

## Prepare data

Data preparation scripts are provided and features in [Kaldi](https://kaldi-asr.org/) format are exepected.

Pre-processed data will consist of hdf files such that features for each recording are stored as datasets of shape [1, nfeat, nframes] and further stored under a group labeled with the speaker ID.

Prepare Kaldi features with data_prep.py. Arguments:

```
--path-to-data        Path to scp files with features
--data-info-path      Path to spk2utt and utt2spk
--spk2utt             Path to spk2utt
--utt2spk             Path to utt2spk
--path-to-more-data   Path to extra scp files with features
--more-data-info-path Path to spk2utt and utt2spk
--more-spk2utt Path   Path to spk2utt
--more-utt2spk Path   Path to utt2spk
--out-path Path       Path to output hdf file
--out-name Path       Output hdf file name
--min-recordings      Minimum number of train recordings for speaker to be included
```

Train and development data hdfs are expected.

## Train a model

Train models with train.py. Arguments:

```
--batch-size          input batch size for training (default: 64)
--epochs              number of epochs to train (default: 500)
--lr                  learning rate (default: 0.001)
--momentum            Momentum paprameter (default: 0.9)
--l2                  Weight decay coefficient (default: 0.00001)
--margin              margin fro triplet loss (default: 0.3)
--lamb                Entropy regularization penalty (default: 0.001)
--swap                Swaps anchor and positive depending on distance to negative example
--patience            Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)
--checkpoint-epoch    epoch to load for checkpointing. If None, training starts from scratch
--checkpoint-path     Path for checkpointing
--pretrained-path     Path for pre trained model
--train-hdf-file      Path to hdf data
--valid-hdf-file      Path to hdf data
--model               {resnet_mfcc,resnet_lstm,resnet_stats,resnet_large,resnet_small}
--delta               Enables extra data channels
--workers             number of data loading workers
--seed                random seed (default: 1)
--save-every          how many epochs to wait before logging training status. Default is 1
--ncoef               number of MFCCs (default: 23)
--latent-size         latent layer dimension (default: 200)
--n-frames            maximum number of frames per utterance (default: 800)
--n-cycles            cycles over speakers list to complete 1 epoch
--valid-n-cycles      cycles over speakers list to complete 1 epoch
--softmax             {none,softmax,am_softmax}
--pretrain            Adds softmax layer for speaker identification and train exclusively with CE minimization
--mine-triplets       Enables distance mining for triplets
--no-cuda             Disables GPU use
--no-cp               Disables checkpointing
--verbose             Verbose is activated if > 0
```

## Scoring test recordings

Scoring with a back-end of choice can be performed on top of embeddings which can be computed with embedd.py.

Arguments:

```
--path-to-data        Path to input data
--cp-path             Path for file containing model
--out-path            Path to output hdf file
--model               {resnet_mfcc,resnet_lstm,resnet_stats,resnet_large}
--latent-size         latent layer dimension (default: 200)
--ncoef               number of MFCCs (default: 23)
--no-cuda             Disables GPU use
```

Simple cosine similarity evaluation of a give set of test trials can be performed with eval.py.
