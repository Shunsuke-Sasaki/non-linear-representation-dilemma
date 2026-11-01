# Shared mapping results (full, align=permutation)

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
| ('alignment_only', 'trained')   | 0.5293 | 0.6527 | 0.5567 | 0.6674 | 0.5077 |
| ('individual', 'random')        | 0.4980 | 0.7321 | 0.4985 | 0.6506 | 0.4967 |
| ('individual', 'trained')       | 0.9996 | 0.9996 | 0.9988 | 0.9993 | 0.9959 |
| ('shared_linear', 'random')     | 0.5008 | 0.4992 | 0.5008 | 0.4992 | 0.4992 |
| ('shared_linear', 'trained')    | 0.6270 | 0.6336 | 0.5556 | 0.6121 | 0.5005 |
| ('shared_nonlinear', 'random')  | 0.4978 | 0.4970 | 0.5008 | 0.5021 | 0.4995 |
| ('shared_nonlinear', 'trained') | 0.5304 | 0.5524 | 0.5065 | 0.5818 | 0.4989 |

## Single-model phi transfer: phi_ref (trained on fold reference only) -> held-out model

| state   |   4287 |   3837 |   9097 |   2635 |   5137 |
|:--------|-------:|-------:|-------:|-------:|-------:|
| random  | 0.4974 | 0.5063 | 0.5008 | 0.4962 | 0.5001 |
| trained | 0.6047 | 0.5403 | 0.5308 | 0.5883 | 0.5285 |

## IIA on the models used for training the shared map

|                                 |   mean |    min |    max |
|:--------------------------------|-------:|-------:|-------:|
| ('alignment_only', 'random')    | 0.4998 | 0.4992 | 0.5008 |
| ('alignment_only', 'trained')   | 0.5930 | 0.5077 | 0.6674 |
| ('shared_linear', 'random')     | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear', 'trained')    | 0.7197 | 0.5293 | 0.8680 |
| ('shared_nonlinear', 'random')  | 0.5202 | 0.4884 | 0.7965 |
| ('shared_nonlinear', 'trained') | 0.9978 | 0.9922 | 0.9996 |

## Summary over the 5 models (folds are not independent; §8.2)

|                                                    |   mean |     min |    max |
|:---------------------------------------------------|-------:|--------:|-------:|
| ('individual', 'trained')                          | 0.9986 |  0.9959 | 0.9996 |
| ('individual', 'random')                           | 0.5752 |  0.4967 | 0.7321 |
| ('individual', 'trained - random')                 | 0.4234 |  0.2675 | 0.5015 |
| ('single_model_transfer(ref)', 'trained')          | 0.5585 |  0.5285 | 0.6047 |
| ('single_model_transfer(ref)', 'random')           | 0.5002 |  0.4962 | 0.5063 |
| ('single_model_transfer(ref)', 'trained - random') | 0.0584 |  0.0284 | 0.1073 |
| ('alignment_only', 'trained')                      | 0.5828 |  0.5077 | 0.6674 |
| ('alignment_only', 'random')                       | 0.4998 |  0.4992 | 0.5008 |
| ('alignment_only', 'trained - random')             | 0.0829 |  0.0085 | 0.1682 |
| ('shared_linear', 'trained')                       | 0.5858 |  0.5005 | 0.6336 |
| ('shared_linear', 'random')                        | 0.4998 |  0.4992 | 0.5008 |
| ('shared_linear', 'trained - random')              | 0.0859 |  0.0013 | 0.1344 |
| ('shared_nonlinear', 'trained')                    | 0.5340 |  0.4989 | 0.5818 |
| ('shared_nonlinear', 'random')                     | 0.4994 |  0.4970 | 0.5021 |
| ('shared_nonlinear', 'trained - random')           | 0.0346 | -0.0006 | 0.0797 |

## Held-out IIA next to the individual IIA of the same model (difference can be negative)

|                                                                    |    4287 |   3837 |    9097 |   2635 |    5137 |   mean |     min |    max |
|:-------------------------------------------------------------------|--------:|-------:|--------:|-------:|--------:|-------:|--------:|-------:|
| ('shared_nonlinear', 'trained', 'individual')                      |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('shared_nonlinear', 'trained', 'held-out')                        |  0.5304 | 0.5524 |  0.5065 | 0.5818 |  0.4989 | 0.5340 |  0.4989 | 0.5818 |
| ('shared_nonlinear', 'trained', 'individual - held-out')           |  0.4692 | 0.4472 |  0.4923 | 0.4175 |  0.4970 | 0.4646 |  0.4175 | 0.4970 |
| ('shared_nonlinear', 'random', 'individual')                       |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('shared_nonlinear', 'random', 'held-out')                         |  0.4978 | 0.4970 |  0.5008 | 0.5021 |  0.4995 | 0.4994 |  0.4970 | 0.5021 |
| ('shared_nonlinear', 'random', 'individual - held-out')            |  0.0002 | 0.2351 | -0.0023 | 0.1485 | -0.0027 | 0.0757 | -0.0027 | 0.2351 |
| ('shared_linear', 'trained', 'individual')                         |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('shared_linear', 'trained', 'held-out')                           |  0.6270 | 0.6336 |  0.5556 | 0.6121 |  0.5005 | 0.5858 |  0.5005 | 0.6336 |
| ('shared_linear', 'trained', 'individual - held-out')              |  0.3725 | 0.3660 |  0.4433 | 0.3872 |  0.4954 | 0.4129 |  0.3660 | 0.4954 |
| ('shared_linear', 'random', 'individual')                          |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('shared_linear', 'random', 'held-out')                            |  0.5008 | 0.4992 |  0.5008 | 0.4992 |  0.4992 | 0.4998 |  0.4992 | 0.5008 |
| ('shared_linear', 'random', 'individual - held-out')               | -0.0028 | 0.2329 | -0.0023 | 0.1514 | -0.0024 | 0.0754 | -0.0028 | 0.2329 |
| ('single_model_transfer(ref)', 'trained', 'individual')            |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('single_model_transfer(ref)', 'trained', 'held-out')              |  0.6047 | 0.5403 |  0.5308 | 0.5883 |  0.5285 | 0.5585 |  0.5285 | 0.6047 |
| ('single_model_transfer(ref)', 'trained', 'individual - held-out') |  0.3948 | 0.4593 |  0.4681 | 0.4110 |  0.4673 | 0.4401 |  0.3948 | 0.4681 |
| ('single_model_transfer(ref)', 'random', 'individual')             |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('single_model_transfer(ref)', 'random', 'held-out')               |  0.4974 | 0.5063 |  0.5008 | 0.4962 |  0.5001 | 0.5002 |  0.4962 | 0.5063 |
| ('single_model_transfer(ref)', 'random', 'individual - held-out')  |  0.0006 | 0.2258 | -0.0023 | 0.1544 | -0.0034 | 0.0750 | -0.0034 | 0.2258 |
| ('alignment_only', 'trained', 'individual')                        |  0.9996 | 0.9996 |  0.9988 | 0.9993 |  0.9959 | 0.9986 |  0.9959 | 0.9996 |
| ('alignment_only', 'trained', 'held-out')                          |  0.5293 | 0.6527 |  0.5567 | 0.6674 |  0.5077 | 0.5828 |  0.5077 | 0.6674 |
| ('alignment_only', 'trained', 'individual - held-out')             |  0.4703 | 0.3469 |  0.4421 | 0.3319 |  0.4882 | 0.4159 |  0.3319 | 0.4882 |
| ('alignment_only', 'random', 'individual')                         |  0.4980 | 0.7321 |  0.4985 | 0.6506 |  0.4967 | 0.5752 |  0.4967 | 0.7321 |
| ('alignment_only', 'random', 'held-out')                           |  0.5008 | 0.4992 |  0.5008 | 0.4992 |  0.4992 | 0.4998 |  0.4992 | 0.5008 |
| ('alignment_only', 'random', 'individual - held-out')              | -0.0028 | 0.2329 | -0.0023 | 0.1514 | -0.0024 | 0.0754 | -0.0028 | 0.2329 |

## Breakdown on the evaluated model (mean over models and inits)

|                                 |   iia_left |   iia_right |   iia_both |   iia_label_changed |   iia_label_unchanged |
|:--------------------------------|-----------:|------------:|-----------:|--------------------:|----------------------:|
| ('alignment_only', 'random')    |     0.4991 |      0.4992 |     0.5011 |              0.4998 |                0.4999 |
| ('alignment_only', 'trained')   |     0.5832 |      0.5796 |     0.5854 |              0.5347 |                0.6319 |
| ('individual', 'random')        |     0.5719 |      0.5753 |     0.5782 |              0.5731 |                0.5774 |
| ('individual', 'trained')       |     0.9988 |      0.9983 |     0.9988 |              0.9988 |                0.9985 |
| ('shared_linear', 'random')     |     0.4991 |      0.4992 |     0.5011 |              0.4998 |                0.4999 |
| ('shared_linear', 'trained')    |     0.5866 |      0.5824 |     0.5882 |              0.5629 |                0.6091 |
| ('shared_nonlinear', 'random')  |     0.5005 |      0.4995 |     0.4984 |              0.4998 |                0.4991 |
| ('shared_nonlinear', 'trained') |     0.5319 |      0.5304 |     0.5395 |              0.5186 |                0.5497 |

## Cross-model swap (§9, evaluation only): base model m, sources from model n

|                                                           |   mean |    min |    max |
|:----------------------------------------------------------|-------:|-------:|-------:|
| ('shared_linear_cross', 'random', 'involves_heldout')     | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear_cross', 'random', 'train_pair')           | 0.4998 | 0.4992 | 0.5008 |
| ('shared_linear_cross', 'trained', 'involves_heldout')    | 0.5197 | 0.4550 | 0.6378 |
| ('shared_linear_cross', 'trained', 'train_pair')          | 0.5145 | 0.4637 | 0.6007 |
| ('shared_nonlinear_cross', 'random', 'involves_heldout')  | 0.4996 | 0.4879 | 0.5109 |
| ('shared_nonlinear_cross', 'random', 'train_pair')        | 0.5010 | 0.4662 | 0.5868 |
| ('shared_nonlinear_cross', 'trained', 'involves_heldout') | 0.5169 | 0.3958 | 0.6556 |
| ('shared_nonlinear_cross', 'trained', 'train_pair')       | 0.5406 | 0.4586 | 0.7027 |

## Alignment cost (permutation) to the reference model

|                     |    2635 |    3837 |    4287 |    5137 |    9097 |
|:--------------------|--------:|--------:|--------:|--------:|--------:|
| ('random', 3837.0)  | 21.0668 |  0.0000 | 21.0896 | 22.4591 | 22.9185 |
| ('random', 4287.0)  | 23.5029 | 21.0896 |  0.0000 | 24.7200 | 21.9800 |
| ('trained', 3837.0) | 16.8220 |  0.0000 | 14.6910 | 19.5451 | 17.9505 |
| ('trained', 4287.0) | 16.3260 | 14.6910 |  0.0000 | 16.6646 | 16.4138 |
