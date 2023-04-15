# meta_train hyperparameters

## Learning rate
--learning_rate 0.1

## C Way
--n_way 20

## K Shot
--n_shot 1

## N Query
--n_query 19

## Meta training episode
--meta_episode 10000

## Embedding Network

| number | input                | operator    | kernel_size | padding   | stride    | output               |
| :------: | :--------------------: | :-----------: | :-----------: | :---------: | :---------: | :--------------------: |
| 1      | [20, 1, 100, 28, 28] | Conv3d      | (3, 3, 3)   | (1, 1, 1) | (1, 1, 1) |  |
|        |                      | BatchNorm3d |             |           |           |  |
|        |                      | ReLU |             |           |           |                      |
|        |                      | MaxPool3d | （4, 2, 2） | (1, 1, 1) | (1, 1, 1) | [20,  8, 25, 15, 15] |
| 2 | [20,  8, 25, 15, 15] | Conv3d | (3, 3, 3) | (1, 1, 1) | (1, 1, 1) |                      |
|        |                      | BatchNorm3d |             |           |           |                      |
|        |                      | ReLU |             |           |           |                      |
|        |                      | MaxPool3d | （4, 2, 2） | (1, 1, 1) | (1, 1, 1) | [20,  16, 6, 8, 8] |
| 3 | [20,  16, 6, 8, 8] | Conv3d | (3, 3, 3) | (1, 1, 1) | (1, 1, 1) |                      |
|        |                      | BatchNorm3d |             |           |           |                      |
|        |                      | ReLU |             |           |           |                      |
|        |                      | MaxPool3d | （4, 2, 2） | (1, 1, 1) | (1, 1, 1) | [20,  64, 2, 5, 5] |
|        |                      |             |             |           |           |                      |



> embedding部分回进入两部分数据，support和query。如果是support，batch维度的数值是20，如果是query，batch维度的数值是380

> support 与 query经过embedding后，得到[20,  64, 2, 5, 5] 与 [380, 64, 2, 5, 5]，两者concatenate后，得到[7600, 256, 5, 5]（这个是使用很复杂的代码（各种view和reshape），其实没搞懂是怎么得到的）

## Relation Network

| number | input                | operator    | kernel_size | padding   | stride    | output               |
| :------: | :--------------------: | :-----------: | :-----------: | :---------: | :---------: | :--------------------: |
| 1 | [7600, 256, 5, 5] | Conv2d | (1, 1) | (1, 1) | (1, 1) | [7600, 128, 5, 5] |
|        |                   | BatchNorm2d |             |         |        |                      |
|        |                      | ReLU |             |           |           |                      |
| 2 | [7600, 128, 5, 5] | Conv2d | (3, 3) | (0, 0) | (1, 1) | [7600, 64, 3, 3] |
|        |                      | BatchNorm2d |             |           |           |                      |
|        |                      | ReLU |             |           |           |                      |
| 3 | [7600, 64, 3, 3] | MaxPool2d | (3, 3) | (0, 0) | (1, 1) | [7600, 64, 1, 1] |
| 4 | [7600, 64, 1, 1] | flatten |             |           |           | [7600, 64] |
| 5 | [7600, 64] | Linear |             |           |           | [7600, 16] |
|        |                      | ReLU |             |           |           |                      |
| 6 | [7600, 16] | Linear |             |           |           | [7600, 1] |
|        |                      | sigmoid |             |  |           |                      |
|        |                      |             |             |           |           |                      |

## Dropout

self.dropout = nn.Dropout(p = 0.5) 

- Relation Network
- in the front of sigmoid

## BatchNorm

- Embedding  Network & Relation Network
- behind every Cond2/3d


## Optimizer
Adam

## StepLR



