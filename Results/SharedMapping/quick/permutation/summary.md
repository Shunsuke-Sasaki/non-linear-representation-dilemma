# Shared mapping results (quick, align=permutation)

## Normal task accuracy (test)

|   target |   random |   trained |
|---------:|---------:|----------:|
|     2635 |   0.4940 |    0.6688 |
|     3837 |   0.5010 |    0.6661 |
|     4287 |   0.4982 |    0.6320 |
|     5137 |   0.5048 |    0.6196 |
|     9097 |   0.4992 |    0.5236 |

## IIA on the evaluated model (individual: own model; others: held-out model of the fold)

Averaged over map initialisations. Columns are model seeds.

|                                 |   4287 |   3837 |   9097 |   2635 |   5137 |
|:--------------------------------|-------:|-------:|-------:|-------:|-------:|
| ('alignment_only', 'random')    | 0.5140 | 0.4860 | 0.5140 | 0.4860 | 0.4860 |
| ('alignment_only', 'trained')   | 0.5115 | 0.5130 | 0.4785 | 0.5340 | 0.5130 |
| ('individual', 'random')        | 0.4965 | 0.4925 | 0.4870 | 0.5135 | 0.4915 |
| ('individual', 'trained')       | 0.6645 | 0.5940 | 0.4965 | 0.6800 | 0.5325 |
| ('shared_linear', 'random')     | 0.5140 | 0.4860 | 0.5140 | 0.4860 | 0.4860 |
| ('shared_linear', 'trained')    | 0.5140 | 0.5025 | 0.5080 | 0.5100 | 0.5120 |
| ('shared_nonlinear', 'random')  | 0.4980 | 0.5060 | 0.5140 | 0.5190 | 0.4860 |
| ('shared_nonlinear', 'trained') | 0.4995 | 0.5090 | 0.4980 | 0.4980 | 0.5035 |

## Single-model phi transfer: phi_ref (trained on fold reference only) -> held-out model

| state   |   4287 |   3837 |   9097 |   2635 |   5137 |
|:--------|-------:|-------:|-------:|-------:|-------:|
| random  | 0.5140 | 0.5125 | 0.5140 | 0.4860 | 0.4860 |
| trained | 0.4945 | 0.4815 | 0.4860 | 0.5035 | 0.4895 |

## IIA on the models used for training the shared map

|                                 |   mean |    min |    max |
|:--------------------------------|-------:|-------:|-------:|
| ('alignment_only', 'random')    | 0.4972 | 0.4860 | 0.5140 |
| ('alignment_only', 'trained')   | 0.5100 | 0.4785 | 0.5340 |
| ('shared_linear', 'random')     | 0.4972 | 0.4860 | 0.5140 |
| ('shared_linear', 'trained')    | 0.5225 | 0.4925 | 0.5560 |
| ('shared_nonlinear', 'random')  | 0.5066 | 0.4820 | 0.5380 |
| ('shared_nonlinear', 'trained') | 0.6071 | 0.4940 | 0.7625 |

## Summary over the 5 models (folds are not independent; §8.2)

|                                                    |    mean |     min |    max |
|:---------------------------------------------------|--------:|--------:|-------:|
| ('individual', 'trained')                          |  0.5935 |  0.4965 | 0.6800 |
| ('individual', 'random')                           |  0.4962 |  0.4870 | 0.5135 |
| ('individual', 'trained - random')                 |  0.0973 |  0.0095 | 0.1680 |
| ('single_model_transfer(ref)', 'trained')          |  0.4910 |  0.4815 | 0.5035 |
| ('single_model_transfer(ref)', 'random')           |  0.5025 |  0.4860 | 0.5140 |
| ('single_model_transfer(ref)', 'trained - random') | -0.0115 | -0.0310 | 0.0175 |
| ('alignment_only', 'trained')                      |  0.5100 |  0.4785 | 0.5340 |
| ('alignment_only', 'random')                       |  0.4972 |  0.4860 | 0.5140 |
| ('alignment_only', 'trained - random')             |  0.0128 | -0.0355 | 0.0480 |
| ('shared_linear', 'trained')                       |  0.5093 |  0.5025 | 0.5140 |
| ('shared_linear', 'random')                        |  0.4972 |  0.4860 | 0.5140 |
| ('shared_linear', 'trained - random')              |  0.0121 | -0.0060 | 0.0260 |
| ('shared_nonlinear', 'trained')                    |  0.5016 |  0.4980 | 0.5090 |
| ('shared_nonlinear', 'random')                     |  0.5046 |  0.4860 | 0.5190 |
| ('shared_nonlinear', 'trained - random')           | -0.0030 | -0.0210 | 0.0175 |

## Held-out IIA next to the individual IIA of the same model (difference can be negative)

|                                                                    |    4287 |    3837 |    9097 |    2635 |   5137 |    mean |     min |    max |
|:-------------------------------------------------------------------|--------:|--------:|--------:|--------:|-------:|--------:|--------:|-------:|
| ('shared_nonlinear', 'trained', 'individual')                      |  0.6645 |  0.5940 |  0.4965 |  0.6800 | 0.5325 |  0.5935 |  0.4965 | 0.6800 |
| ('shared_nonlinear', 'trained', 'held-out')                        |  0.4995 |  0.5090 |  0.4980 |  0.4980 | 0.5035 |  0.5016 |  0.4980 | 0.5090 |
| ('shared_nonlinear', 'trained', 'individual - held-out')           |  0.1650 |  0.0850 | -0.0015 |  0.1820 | 0.0290 |  0.0919 | -0.0015 | 0.1820 |
| ('shared_nonlinear', 'random', 'individual')                       |  0.4965 |  0.4925 |  0.4870 |  0.5135 | 0.4915 |  0.4962 |  0.4870 | 0.5135 |
| ('shared_nonlinear', 'random', 'held-out')                         |  0.4980 |  0.5060 |  0.5140 |  0.5190 | 0.4860 |  0.5046 |  0.4860 | 0.5190 |
| ('shared_nonlinear', 'random', 'individual - held-out')            | -0.0015 | -0.0135 | -0.0270 | -0.0055 | 0.0055 | -0.0084 | -0.0270 | 0.0055 |
| ('shared_linear', 'trained', 'individual')                         |  0.6645 |  0.5940 |  0.4965 |  0.6800 | 0.5325 |  0.5935 |  0.4965 | 0.6800 |
| ('shared_linear', 'trained', 'held-out')                           |  0.5140 |  0.5025 |  0.5080 |  0.5100 | 0.5120 |  0.5093 |  0.5025 | 0.5140 |
| ('shared_linear', 'trained', 'individual - held-out')              |  0.1505 |  0.0915 | -0.0115 |  0.1700 | 0.0205 |  0.0842 | -0.0115 | 0.1700 |
| ('shared_linear', 'random', 'individual')                          |  0.4965 |  0.4925 |  0.4870 |  0.5135 | 0.4915 |  0.4962 |  0.4870 | 0.5135 |
| ('shared_linear', 'random', 'held-out')                            |  0.5140 |  0.4860 |  0.5140 |  0.4860 | 0.4860 |  0.4972 |  0.4860 | 0.5140 |
| ('shared_linear', 'random', 'individual - held-out')               | -0.0175 |  0.0065 | -0.0270 |  0.0275 | 0.0055 | -0.0010 | -0.0270 | 0.0275 |
| ('single_model_transfer(ref)', 'trained', 'individual')            |  0.6645 |  0.5940 |  0.4965 |  0.6800 | 0.5325 |  0.5935 |  0.4965 | 0.6800 |
| ('single_model_transfer(ref)', 'trained', 'held-out')              |  0.4945 |  0.4815 |  0.4860 |  0.5035 | 0.4895 |  0.4910 |  0.4815 | 0.5035 |
| ('single_model_transfer(ref)', 'trained', 'individual - held-out') |  0.1700 |  0.1125 |  0.0105 |  0.1765 | 0.0430 |  0.1025 |  0.0105 | 0.1765 |
| ('single_model_transfer(ref)', 'random', 'individual')             |  0.4965 |  0.4925 |  0.4870 |  0.5135 | 0.4915 |  0.4962 |  0.4870 | 0.5135 |
| ('single_model_transfer(ref)', 'random', 'held-out')               |  0.5140 |  0.5125 |  0.5140 |  0.4860 | 0.4860 |  0.5025 |  0.4860 | 0.5140 |
| ('single_model_transfer(ref)', 'random', 'individual - held-out')  | -0.0175 | -0.0200 | -0.0270 |  0.0275 | 0.0055 | -0.0063 | -0.0270 | 0.0275 |
| ('alignment_only', 'trained', 'individual')                        |  0.6645 |  0.5940 |  0.4965 |  0.6800 | 0.5325 |  0.5935 |  0.4965 | 0.6800 |
| ('alignment_only', 'trained', 'held-out')                          |  0.5115 |  0.5130 |  0.4785 |  0.5340 | 0.5130 |  0.5100 |  0.4785 | 0.5340 |
| ('alignment_only', 'trained', 'individual - held-out')             |  0.1530 |  0.0810 |  0.0180 |  0.1460 | 0.0195 |  0.0835 |  0.0180 | 0.1530 |
| ('alignment_only', 'random', 'individual')                         |  0.4965 |  0.4925 |  0.4870 |  0.5135 | 0.4915 |  0.4962 |  0.4870 | 0.5135 |
| ('alignment_only', 'random', 'held-out')                           |  0.5140 |  0.4860 |  0.5140 |  0.4860 | 0.4860 |  0.4972 |  0.4860 | 0.5140 |
| ('alignment_only', 'random', 'individual - held-out')              | -0.0175 |  0.0065 | -0.0270 |  0.0275 | 0.0055 | -0.0010 | -0.0270 | 0.0275 |

## Breakdown on the evaluated model (mean over models and inits)

|                                 |   iia_left |   iia_right |   iia_both |   iia_label_changed |   iia_label_unchanged |
|:--------------------------------|-----------:|------------:|-----------:|--------------------:|----------------------:|
| ('alignment_only', 'random')    |     0.4971 |      0.4988 |     0.4957 |              0.4954 |                0.4990 |
| ('alignment_only', 'trained')   |     0.4995 |      0.5178 |     0.5126 |              0.4754 |                0.5447 |
| ('individual', 'random')        |     0.4866 |      0.5012 |     0.5007 |              0.4926 |                0.4998 |
| ('individual', 'trained')       |     0.5937 |      0.5780 |     0.6086 |              0.5758 |                0.6112 |
| ('shared_linear', 'random')     |     0.4971 |      0.4988 |     0.4957 |              0.4954 |                0.4990 |
| ('shared_linear', 'trained')    |     0.5014 |      0.5160 |     0.5105 |              0.4804 |                0.5383 |
| ('shared_nonlinear', 'random')  |     0.4950 |      0.5027 |     0.5159 |              0.5062 |                0.5030 |
| ('shared_nonlinear', 'trained') |     0.4995 |      0.4976 |     0.5076 |              0.4940 |                0.5092 |

## Cross-model swap (§9, evaluation only): base model m, sources from model n

|                                                           |   mean |    min |    max |
|:----------------------------------------------------------|-------:|-------:|-------:|
| ('shared_linear_cross', 'random', 'involves_heldout')     | 0.4972 | 0.4860 | 0.5140 |
| ('shared_linear_cross', 'random', 'train_pair')           | 0.4972 | 0.4860 | 0.5140 |
| ('shared_linear_cross', 'trained', 'involves_heldout')    | 0.5040 | 0.4795 | 0.5360 |
| ('shared_linear_cross', 'trained', 'train_pair')          | 0.5029 | 0.4730 | 0.5255 |
| ('shared_nonlinear_cross', 'random', 'involves_heldout')  | 0.5035 | 0.4820 | 0.5225 |
| ('shared_nonlinear_cross', 'random', 'train_pair')        | 0.5037 | 0.4800 | 0.5310 |
| ('shared_nonlinear_cross', 'trained', 'involves_heldout') | 0.5032 | 0.4700 | 0.5350 |
| ('shared_nonlinear_cross', 'trained', 'train_pair')       | 0.5139 | 0.4715 | 0.5885 |

## Alignment cost (permutation) to the reference model

|                     |    2635 |    3837 |    4287 |    5137 |    9097 |
|:--------------------|--------:|--------:|--------:|--------:|--------:|
| ('random', 3837.0)  | 21.2830 |  0.0000 | 21.1573 | 22.3589 | 22.7283 |
| ('random', 4287.0)  | 23.2873 | 21.1573 |  0.0000 | 24.6832 | 21.8938 |
| ('trained', 3837.0) | 22.3240 |  0.0000 | 21.2149 | 21.4337 | 23.1668 |
| ('trained', 4287.0) | 21.9580 | 21.2149 |  0.0000 | 22.9621 | 22.6745 |
