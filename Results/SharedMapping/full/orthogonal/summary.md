# Shared mapping results (full, align=orthogonal)

## Normal task accuracy (test)

|   target |   random |   trained |
|---------:|---------:|----------:|
|     2635 |   0.4980 |    1.0000 |
|     3837 |   0.4831 |    0.9997 |
|     4287 |   0.4967 |    1.0000 |
|     5137 |   0.5005 |    0.9993 |
|     9097 |   0.4986 |    0.9996 |

## IIA on the evaluated model (individual: own model; others: held-out model of the fold)

Averaged over map initialisations. Columns are model seeds.

|                                 |   4287 |   3837 |   9097 |   2635 |   5137 |
|:--------------------------------|-------:|-------:|-------:|-------:|-------:|
| ('alignment_only', 'random')    | 0.5008 | 0.4992 | 0.5008 | 0.4992 | 0.4992 |
| ('alignment_only', 'trained')   | 0.5227 | 0.5979 | 0.5257 | 0.5483 | 0.5502 |
| ('individual', 'random')        | 0.4980 | 0.7321 | 0.4985 | 0.6506 | 0.4967 |
| ('individual', 'trained')       | 0.9996 | 0.9996 | 0.9988 | 0.9993 | 0.9959 |
| ('shared_linear', 'random')     | 0.5008 | 0.4992 | 0.5008 | 0.4992 | 0.4992 |
| ('shared_linear', 'trained')    | 0.6638 | 0.6312 | 0.5499 | 0.6874 | 0.5227 |
| ('shared_nonlinear', 'random')  | 0.4996 | 0.5013 | 0.5008 | 0.4980 | 0.4992 |
| ('shared_nonlinear', 'trained') | 0.5821 | 0.6613 | 0.5859 | 0.5711 | 0.4990 |

## Single-model phi transfer: phi_ref (trained on fold reference only) -> held-out model

| state   |   4287 |   3837 |   9097 |   2635 |   5137 |
|:--------|-------:|-------:|-------:|-------:|-------:|
| random  | 0.4989 | 0.5062 | 0.5008 | 0.4981 | 0.5035 |
| trained | 0.6553 | 0.6541 | 0.5666 | 0.6714 | 0.5303 |

## IIA on the models used for training the shared map

|                                 |   mean |    min |    max |
|:--------------------------------|-------:|-------:|-------:|
| ('alignment_only', 'random')    | 0.4998 | 0.4992 | 0.5008 |
| ('alignment_only', 'trained')   | 0.5677 | 0.5038 | 0.6478 |
| ('shared_linear', 'random')     | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear', 'trained')    | 0.7047 | 0.5418 | 0.8264 |
| ('shared_nonlinear', 'random')  | 0.5470 | 0.4858 | 0.7786 |
| ('shared_nonlinear', 'trained') | 0.9979 | 0.9920 | 0.9995 |

## Summary over the 5 models (folds are not independent; §8.2)

|                                                    |   mean |     min |    max |
|:---------------------------------------------------|-------:|--------:|-------:|
| ('individual', 'trained')                          | 0.9986 |  0.9959 | 0.9996 |
| ('individual', 'random')                           | 0.5752 |  0.4967 | 0.7321 |
| ('individual', 'trained - random')                 | 0.4234 |  0.2675 | 0.5015 |
| ('single_model_transfer(ref)', 'trained')          | 0.6156 |  0.5303 | 0.6714 |
| ('single_model_transfer(ref)', 'random')           | 0.5015 |  0.4981 | 0.5062 |
| ('single_model_transfer(ref)', 'trained - random') | 0.1140 |  0.0267 | 0.1733 |
| ('alignment_only', 'trained')                      | 0.5490 |  0.5227 | 0.5979 |
| ('alignment_only', 'random')                       | 0.4998 |  0.4992 | 0.5008 |
| ('alignment_only', 'trained - random')             | 0.0491 |  0.0219 | 0.0987 |
| ('shared_linear', 'trained')                       | 0.6110 |  0.5227 | 0.6874 |
| ('shared_linear', 'random')                        | 0.4998 |  0.4992 | 0.5008 |
| ('shared_linear', 'trained - random')              | 0.1111 |  0.0235 | 0.1882 |
| ('shared_nonlinear', 'trained')                    | 0.5799 |  0.4990 | 0.6613 |
| ('shared_nonlinear', 'random')                     | 0.4998 |  0.4980 | 0.5013 |
| ('shared_nonlinear', 'trained - random')           | 0.0801 | -0.0001 | 0.1600 |

## Held-out IIA next to the individual IIA of the same model (difference can be negative)

|                                                                    |    4287 |   3837 |    9097 |   2635 |    5137 |   mean |     min |    max |
|:-------------------------------------------------------------------|--------:|-------:|--------:|-------:|--------:|-------:|--------:|-------:|
| ('shared_nonlinear', 'trained', 'individual')                      |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('shared_nonlinear', 'trained', 'held-out')                        |  0.5821 | 0.6613 |  0.5859 | 0.5711 |  0.4990 | 0.5799 |  0.4990 | 0.6613 |
| ('shared_nonlinear', 'trained', 'individual - held-out')           |  0.4175 | 0.3383 |  0.4129 | 0.4282 |  0.4968 | 0.4187 |  0.3383 | 0.4968 |
| ('shared_nonlinear', 'random', 'individual')                       |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('shared_nonlinear', 'random', 'held-out')                         |  0.4996 | 0.5013 |  0.5008 | 0.4980 |  0.4992 | 0.4998 |  0.4980 | 0.5013 |
| ('shared_nonlinear', 'random', 'individual - held-out')            | -0.0015 | 0.2307 | -0.0023 | 0.1526 | -0.0024 | 0.0754 | -0.0024 | 0.2307 |
| ('shared_linear', 'trained', 'individual')                         |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('shared_linear', 'trained', 'held-out')                           |  0.6638 | 0.6312 |  0.5499 | 0.6874 |  0.5227 | 0.6110 |  0.5227 | 0.6874 |
| ('shared_linear', 'trained', 'individual - held-out')              |  0.3358 | 0.3684 |  0.4489 | 0.3119 |  0.4732 | 0.3877 |  0.3119 | 0.4732 |
| ('shared_linear', 'random', 'individual')                          |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('shared_linear', 'random', 'held-out')                            |  0.5008 | 0.4992 |  0.5008 | 0.4992 |  0.4992 | 0.4998 |  0.4992 | 0.5008 |
| ('shared_linear', 'random', 'individual - held-out')               | -0.0028 | 0.2329 | -0.0023 | 0.1514 | -0.0024 | 0.0754 | -0.0028 | 0.2329 |
| ('single_model_transfer(ref)', 'trained', 'individual')            |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('single_model_transfer(ref)', 'trained', 'held-out')              |  0.6553 | 0.6541 |  0.5666 | 0.6714 |  0.5303 | 0.6156 |  0.5303 | 0.6714 |
| ('single_model_transfer(ref)', 'trained', 'individual - held-out') |  0.3442 | 0.3455 |  0.4322 | 0.3279 |  0.4656 | 0.3831 |  0.3279 | 0.4656 |
| ('single_model_transfer(ref)', 'random', 'individual')             |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('single_model_transfer(ref)', 'random', 'held-out')               |  0.4989 | 0.5062 |  0.5008 | 0.4981 |  0.5035 | 0.5015 |  0.4981 | 0.5062 |
| ('single_model_transfer(ref)', 'random', 'individual - held-out')  | -0.0008 | 0.2259 | -0.0023 | 0.1525 | -0.0068 | 0.0737 | -0.0068 | 0.2259 |
| ('alignment_only', 'trained', 'individual')                        |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('alignment_only', 'trained', 'held-out')                          |  0.5227 | 0.5979 |  0.5257 | 0.5483 |  0.5502 | 0.5490 |  0.5227 | 0.5979 |
| ('alignment_only', 'trained', 'individual - held-out')             |  0.4769 | 0.4017 |  0.4732 | 0.4510 |  0.4457 | 0.4497 |  0.4017 | 0.4769 |
| ('alignment_only', 'random', 'individual')                         |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('alignment_only', 'random', 'held-out')                           |  0.5008 | 0.4992 |  0.5008 | 0.4992 |  0.4992 | 0.4998 |  0.4992 | 0.5008 |
| ('alignment_only', 'random', 'individual - held-out')              | -0.0028 | 0.2329 | -0.0023 | 0.1514 | -0.0024 | 0.0754 | -0.0028 | 0.2329 |

## Breakdown on the evaluated model (mean over models and inits)

|                                 |   iia_left |   iia_right |   iia_both |   iia_label_changed |   iia_label_unchanged |
|:--------------------------------|-----------:|------------:|-----------:|--------------------:|----------------------:|
| ('alignment_only', 'random')    |     0.4991 |      0.4992 |     0.5011 |              0.4998 |                0.4999 |
| ('alignment_only', 'trained')   |     0.5523 |      0.5457 |     0.5490 |              0.5186 |                0.5799 |
| ('individual', 'random')        |     0.5719 |      0.5753 |     0.5782 |              0.5731 |                0.5774 |
| ('individual', 'trained')       |     0.9988 |      0.9983 |     0.9988 |              0.9988 |                0.9985 |
| ('shared_linear', 'random')     |     0.4991 |      0.4992 |     0.5011 |              0.4998 |                0.4999 |
| ('shared_linear', 'trained')    |     0.6170 |      0.6067 |     0.6094 |              0.5927 |                0.6297 |
| ('shared_nonlinear', 'random')  |     0.4992 |      0.5000 |     0.5001 |              0.4999 |                0.4997 |
| ('shared_nonlinear', 'trained') |     0.5824 |      0.5744 |     0.5829 |              0.5639 |                0.5963 |

## Cross-model swap (§9, evaluation only): base model m, sources from model n

|                                                           |   mean |    min |    max |
|:----------------------------------------------------------|-------:|-------:|-------:|
| ('shared_linear_cross', 'random', 'involves_heldout')     | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear_cross', 'random', 'train_pair')           | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear_cross', 'trained', 'involves_heldout')    | 0.5670 | 0.4641 | 0.7756 |
| ('shared_linear_cross', 'trained', 'train_pair')          | 0.5806 | 0.4386 | 0.7969 |
| ('shared_nonlinear_cross', 'random', 'involves_heldout')  | 0.5041 | 0.4918 | 0.5581 |
| ('shared_nonlinear_cross', 'random', 'train_pair')        | 0.5261 | 0.4869 | 0.7258 |
| ('shared_nonlinear_cross', 'trained', 'involves_heldout') | 0.6003 | 0.4618 | 0.8638 |
| ('shared_nonlinear_cross', 'trained', 'train_pair')       | 0.6804 | 0.4647 | 0.9124 |

## Alignment cost (orthogonal) to the reference model

|                     |   2635 |   3837 |   4287 |   5137 |   9097 |
|:--------------------|-------:|-------:|-------:|-------:|-------:|
| ('random', 3837.0)  | 0.0089 | 0.0000 | 0.0103 | 0.0101 | 0.0111 |
| ('random', 4287.0)  | 0.0121 | 0.0103 | 0.0000 | 0.0113 | 0.0114 |
| ('trained', 3837.0) | 0.0268 | 0.0000 | 0.0266 | 0.0527 | 0.0440 |
| ('trained', 4287.0) | 0.0290 | 0.0266 | 0.0000 | 0.0488 | 0.0399 |
