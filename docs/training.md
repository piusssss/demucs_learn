# Training (Hybrid) Demucs

## Install all the dependencies

You should install all the dependencies either with either Anaconda (using the env file `environment-cuda.yml` )
or `pip`, with `requirements.txt`.

## Datasets

### MusDB HQ

Note that we do not support MusDB non HQ training anymore.
Get the [Musdb HQ](https://zenodo.org/record/3338373) dataset, and update the path to it in two places:
- The `dset.musdb` key inside `conf/config.yaml`.
- The variable `MUSDB_PATH` inside `tools/automix.py`.

### Create the fine tuning datasets

**This is only for the MDX 2021 competition models**

I use a fine tuning on a dataset crafted by remixing songs in a musically plausible way.
The automix script will make sure that BPM, first beat and pitches are aligned.
In the file `tools/automix.py`, edit `OUTPATH` to suit your setup, as well as the `MUSDB_PATH`
to point to your copy of MusDB HQ. Then run

```bash
export NUMBA_NUM_THREADS=1; python3 -m tools.automix
```

**Important:** the script will show many errors, those are normals. They just indicate when two stems
 do not batch due to BPM or music scale difference.

Finally, edit the file `conf/dset/auto_mus.yaml` and replace `dset.wav` to the value of `OUTPATH`.

If you have a custom dataset, you can also uncomment the lines `dset2 = ...` and
`dset3 = ...` to add your custom wav data and the test set of MusDB for Track B models.
You can then replace the paths in `conf/dset/auto_extra.yaml`, `conf/dset/auto_extra_test.yaml`
and `conf/dset/aetl.yaml` (this last one was using 10 mixes instead of 6 for each song).

### Dataset metadata cache

Datasets are scanned the first time they are used to determine the files and their durations.
If you change a dataset and need a rescan, just delete the `metadata` folder.

## A short intro to Dora

I use [Dora][dora] for all the of experiments (XPs) management. You should have a look at the Dora README
to learn about the tool. Here is a quick summary of what to know:

- An XP is a unique set of hyper-parameters with a given signature. The signature is a hash of
    those hyper-parameters. I will always refer to an XP with its signature, e.g. `9357e12e`.
    We will see after that you can retrieve the hyper-params and re-rerun it in a single command.
- In fact, the hash is defined as a delta between the base config and the one obtained with
    the config overrides you passed from the command line.
    **This means you must never change the `conf/**.yaml` files directly.**,
    except for editing things like paths. Changing the default values in the config files means
    the XP signature won't reflect that change, and wrong checkpoints might be reused.
    I know, this is annoying, but the reason is that otherwise, any change to the config file would
    mean that all XPs ran so far would see their signature change.

### Dora commands

Run `tar xvf outputs.tar.gz`. This will initialize the Dora XP repository, so that Dora knows
which hyper-params match the signature like `9357e12e`. Once you have done that, you should be able
to run the following:

```bash
dora info -f 81de367c  # this will show the hyper-parameter used by a specific XP.
                       # Be careful some overrides might present twice, and the right most one
                       # will give you the right value for it.
dora run -d -f 81de367c   # run an XP with the hyper-parameters from XP 81de367c.
                          # `-d` is for distributed, it will use all available GPUs.
dora run -d -f 81de367c hdemucs.channels=32  # start from the config of XP 81de367c but change some hyper-params.
                                             # This will give you a new XP with a new signature (here 3fe9c332).
```

An XP runs from a specific folder based on its signature, by default under the `outputs/` folder.
You can safely interrupt a training and resume it, it will reuse any existing checkpoint, as it will
reuse the same folder.
If you made some change to the code and need to ignore a previous checkpoint you can use `dora run --clear [RUN ARGS]`.

If you have a Slurm cluster, you can also use the `dora grid` command, e.g. `dora grid mdx`.
Please refer to the [Dora documentation][dora] for more information.

## Hyper parameters

Have a look at [conf/config.yaml](../conf/config.yaml) for a list of all the hyper-parameters you can override.
If you are not familiar with [Hydra](https://github.com/facebookresearch/hydra), go checkout their page
to be familiar with how to provide overrides for your trainings.


## Model architecture

A number of architectures are supported. You can select one with `model=NAME`, and have a look
in [conf/config.yaml'(../conf/config.yaml) for each architecture specific hyperparams.
Those specific params will be always prefixed with the architecture name when passing the override
from the command line or in grid files. Here is the list of models:

- demucs: original time-only Demucs.
- hdemucs: Hybrid Demucs (v3).
- torch_hdemucs: Same as Hybrid Demucs, but using [torchaudio official implementation](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html).
- htdemucs: Hybrid Transformer Demucs (v4).

### Storing config in files

As mentioned earlier, you should never change the base config files. However, you can use Hydra config groups
in order to store variants you often use. If you want to create a new variant combining multiple hyper-params,
copy the file `conf/variant/example.yaml` to `conf/variant/my_variant.yaml`, and then you can use it with

```bash
dora run -d variant=my_variant
```

Once you have created this file, you should not edit it once you have started training models with it.


## Fine tuning

If a first model is trained, you can fine tune it with other settings (e.g. automix dataset) with

```bash
dora run -d -f 81de367c continue_from=81de367c dset=auto_mus variant=finetune
````

Note that you need both `-f 81de367c` and `continue_from=81de367c`. The first one indicates
that the hyper-params of `81de367c` should be used as a starting point for the config.
The second indicates that the weights from `81de367c` should be used as a starting point for the solver.


## Model evaluation

Your model will be evaluated automatically with the new SDR definition from MDX every 20 epochs.
Old style SDR (which is quite slow) will only happen at the end of training.

## Model Export


In order to use your models with other commands (such as the `demucs` command for separation) you must
export it. For that run

```bash
python3 -m tools.export 9357e12e [OTHER SIGS ...]  # replace with the appropriate signatures.
```

The models will be stored under `release_models/`. You can use them with the `demucs` separation command with the following flags:
```bash
demucs --repo ./release_models -n 9357e12e my_track.mp3
```

### Bag of models

If you want to combine multiple models, potentially with different weights for each source, you can copy
`demucs/remote/mdx.yaml` to `./release_models/my_bag.yaml`. You can then edit the list of models (all models used should have been exported first) and the weights per source and model (list of list, outer list is over models, inner list is over sources). You can then use your bag of model as

```bash
demucs --repo ./release_models -n my_bag my_track.mp3
```

## Model evaluation

You can evaluate any pre-trained model or bag of models using the following command:
```bash
python3 -m tools.test_pretrained -n NAME_OF_MODEL [EXTRA ARGS]
```
where `NAME_OF_MODEL` is either the name of the bag (e.g. `mdx`, `repro_mdx_a`),
or a single Dora signature of one of the model of the bags. You can pass `EXTRA ARGS` to customize
the test options, like the number of random shifts (e.g. `test.shifts=2`). This will compute the old-style
SDR and can take quite  bit of time.

For custom models that were trained locally, you will need to indicate that you wish
to use the local model repositories, with the `--repo ./release_models` flag, e.g.,
```bash
python3 -m tools.test_pretrained --repo ./release_models -n my_bag
```


## API to retrieve the model

You can retrieve officially released models in Python using the following API:
```python
from demucs import pretrained
from demucs.apply import apply_model
bag = pretrained.get_model('htdemucs')    # for a bag of models or a named model
                                          # (which is just a bag with 1 model).
model = pretrained.get_model('955717e8')  # using the signature for single models.

bag.models                       # list of individual models
stems = apply_model(model, mix)  # apply the model to the given mix.
```

## Model Zoo

### Hybrid Transformer Demucs

The configuration for the Hybrid Transformer models are available in:

```shell
dora grid mmi --dry_run --init
dora grid mmi_ft --dry_run --init  # fined tuned on each sources.
```

We release in particular `955717e8`, Hybrid Transformer Demucs using 5 layers, 512 channels, 10 seconds training segment length. We also release its fine tuned version, with one model
for each source `f7e0c4bc`, `d12395a8`, `92cfc3b6`, `04573f0d` (drums, bass, other, vocals).
The model `955717e8` is also named `htdemucs`, while the bag of models is provided
as `htdemucs_ft`.

We also release `75fc33f5`, a regular Hybrid Demucs trained on the same dataset,
available as `hdemucs_mmi`.



### Models from the MDX Competition 2021

  
Here is a short descriptions of the models used for the MDX submission, either Track A (MusDB HQ only)
or Track B (extra training data allowed). Training happen in two stage, with the second stage
being the fine tunining on the automix generated dataset.
All the fine tuned models are available on our AWS repository
(you can retrieve it with `demucs.pretrained.get_model(SIG)`). The bag of models are available
by doing `demucs.pretrained.get_model(NAME)` with `NAME` begin either `mdx` (for Track A) or `mdx_extra`
(for Track B).

#### Track A

The 4 models are:

- `0d19c1c6`: fine-tuned on automix dataset from `9357e12e`
- `7ecf8ec1`: fine-tuned on automix dataset from `e312f349`
- `c511e2ab`: fine-tuned on automix dataset from `81de367c`
- `7d865c68`: fine-tuned on automix dataset from `80a68df8`

The 4 initial models (before fine tuning are):

- `9357e12e`: 64ch time domain only improved Demucs, with new residual branches, group norm,
  and singular value penalty.
- `e312f349`: 64ch time domain only improved, with new residual branches, group norm,
  and singular value penalty, trained with a loss that focus only on drums and bass.
- `81de367c`: 48ch hybrid model , with residual branches, group norm,
  singular value penalty penalty and amplitude spectrogram.
- `80a68df8`: same as b5559babb but using CaC and different
  random seed, as well different weigths per frequency bands in outermost layers.

The hybrid models are combined with equal weights for all sources except for the bass.
`0d19c1c6` (time domain) is used for both drums and bass. `7ecf8ec1` is used only for the bass.

You can see all the hyper parameters at once with (one common line for all common hyper params, and then only shows
the hyper parameters that differs), along with the DiffQ variants that are used for the `mdx_q` models:
```
dora grid mdx --dry_run --init
dora grid mdx --dry_run --init
```

#### Track B

- `e51eebcc`
- `a1d90b5c`
- `5d2d6c55`
- `cfa93e08`

All the models are 48ch hybrid demucs with different random seeds. Two of them
are using CaC, and two are using amplitude spectrograms with masking.
All the models are combined with equal weights for all sources.

Things are a bit messy for Track B, there was a lot of fine tuning
over different datasets. I won't describe the entire genealogy of models here,
but all the information can be accessed with the `dora info -f SIG` command.

Similarly you can do (those will contain a few extra lines, for training without the MusDB test set as training, and extra DiffQ XPs):
```
dora grid mdx_extra --dry_run --init
```

### Reproducibility and Ablation

I updated the paper to report numbers with a more homogeneous setup than the one used for the competition.
On MusDB HQ, I still need to use a combination of time only and hybrid models to achieve the best performance.
The experiments are provided in the grids [repro.py](../demucs/grids/repro.py) and
[repro_ft._py](../demucs/grids/repro_ft.py) for the fine tuning on the realistic mix datasets.

The new bag of models reaches an SDR of 7.64 (vs. 7.68 for the original track A model). It uses
2 time only models trained with residual branches, local attention and the SVD penalty,
along with 2 hybrid models, with the same features, and using CaC representation.
We average the performance of all the models with the same weight over all sources, unlike
what was done for the original track A model. We trained for 600 epochs, against 360 before.

The new bag of model is available as part of the pretrained model as `repro_mdx_a`.
The time only bag is named `repro_mdx_a_time_only`, and the hybrid only `repro_mdx_a_hybrid_only`.
Checkout the paper for more information on the training.

[dora]: https://github.com/facebookresearch/dora

好的，这是 `docs/training.md` 文件的完整中文翻译。

---

# 训练 (混合) Demucs

## 安装所有依赖

您应该使用 Anaconda（通过 `environment-cuda.yml` 环境文件）或 `pip`（通过 `requirements.txt` 文件）来安装所有依赖项。

## 数据集

### MusDB HQ

请注意，我们不再支持使用非 HQ 版本的 MusDB 进行训练。
请获取 [Musdb HQ](https://zenodo.org/record/3338373) 数据集，并在以下两个位置更新其路径：
- `conf/config.yaml` 文件中的 `dset.musdb` 键。
- `tools/automix.py` 文件中的 `MUSDB_PATH` 变量。

### 创建微调数据集

**此部分仅适用于 MDX 2021 竞赛模型**

我使用一个通过音乐上合理的方式混音歌曲而制作的数据集进行微调。
`automix` 脚本将确保 BPM（每分钟节拍数）、第一拍和音高都对齐。
在 `tools/automix.py` 文件中，编辑 `OUTPATH` 以适应您的设置，同时将 `MUSDB_PATH` 指向您的 MusDB HQ 副本。然后运行：

```bash
export NUMBA_NUM_THREADS=1; python3 -m tools.automix
```

**重要提示：** 该脚本会显示许多错误，这些都是正常的。它们仅表示两个音轨由于 BPM 或音阶差异而无法匹配。

最后，编辑 `conf/dset/auto_mus.yaml` 文件，并将 `dset.wav` 的值替换为您的 `OUTPATH`。

如果您有自定义数据集，您也可以取消注释 `dset2 = ...` 和 `dset3 = ...` 这几行，以添加您的自定义 wav 数据和用于 B 赛道模型的 MusDB 测试集。
然后，您可以替换 `conf/dset/auto_extra.yaml`、`conf/dset/auto_extra_test.yaml` 和 `conf/dset/aetl.yaml` 中的路径（最后一个文件对每首歌曲使用了10个混音而不是6个）。

### 数据集元数据缓存

数据集在首次使用时会被扫描，以确定文件及其时长。
如果您更改了数据集并需要重新扫描，只需删除 `metadata` 文件夹即可。

## Dora 简介

我使用 [Dora][dora] 来管理所有的实验（XPs）。您应该查看 Dora 的 README 来学习这个工具。以下是需要了解的快速摘要：

- 一个 XP（实验）是一组具有给定签名的唯一超参数集合。签名是这些超参数的哈希值。我将始终使用其签名来引用一个 XP，例如 `9357e12e`。
  我们稍后会看到，您可以通过单个命令检索超参数并重新运行它。
- 事实上，哈希值被定义为基础配置与您从命令行传递的配置覆盖所获得的配置之间的增量。
  **这意味着您绝不能直接更改 `conf/**.yaml` 文件**，除非是编辑像路径这样的东西。更改配置文件中的默认值意味着 XP 签名将不会反映该更改，并且可能会重用错误的检查点。
  我知道这很烦人，但原因在于，否则对配置文件的任何更改都将意味着迄今为止运行的所有 XP 的签名都会改变。

### Dora 命令

运行 `tar xvf outputs.tar.gz`。这将初始化 Dora XP 仓库，以便 Dora 知道哪些超参数匹配像 `9357e12e` 这样的签名。完成此操作后，您应该能够运行以下命令：

```bash
dora info -f 81de367c  # 这将显示特定 XP 使用的超参数。
                       # 注意，某些覆盖项可能会出现两次，最右边的一个将为您提供其正确的值。
dora run -d -f 81de367c   # 使用 XP 81de367c 的超参数运行一个 XP。
                          # `-d` 代表分布式，它将使用所有可用的 GPU。
dora run -d -f 81de367c hdemucs.channels=32  # 从 XP 81de367c 的配置开始，但更改一些超参数。
                                             # 这将给您一个具有新签名的新 XP（此处为 3fe9c332）。
```

一个 XP 会从一个基于其签名的特定文件夹中运行，默认位于 `outputs/` 文件夹下。
您可以安全地中断训练并恢复它，它将重用任何现有的检查点，因为它会重用同一个文件夹。
如果您对代码进行了一些更改并需要忽略之前的检查点，您可以使用 `dora run --clear [RUN ARGS]`。

如果您有 Slurm 集群，您还可以使用 `dora grid` 命令，例如 `dora grid mdx`。
请参阅 [Dora 文档][dora] 获取更多信息。

## 超参数

请查看 [conf/config.yaml](../conf/config.yaml) 以获取可以覆盖的所有超参数的列表。
如果您不熟悉 [Hydra](https://github.com/facebookresearch/hydra)，请查看他们的页面以熟悉如何为您的训练提供覆盖项。

## 模型架构

支持多种架构。您可以使用 `model=NAME` 选择一个，并在 [conf/config.yaml](../conf/config.yaml) 中查看每种架构特定的超参数。
当从命令行或在网格文件中传递覆盖项时，这些特定参数将始终以架构名称为前缀。以下是模型列表：

- demucs: 原始的纯时域 Demucs。
- hdemucs: 混合 Demucs (v3)。
- torch_hdemucs: 与混合 Demucs 相同，但使用 [torchaudio 官方实现](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html)。
- htdemucs: 混合 Transformer Demucs (v4)。

### 将配置存储在文件中

如前所述，您不应更改基础配置文件。但是，您可以使用 Hydra 配置组来存储您经常使用的变体。如果您想创建一个结合多个超参数的新变体，
请将 `conf/variant/example.yaml` 文件复制到 `conf/variant/my_variant.yaml`，然后您可以使用以下命令来使用它：

```bash
dora run -d variant=my_variant
```

一旦创建了这个文件，在您开始用它训练模型后就不应再编辑它。

## 微调

如果第一个模型已经训练好，您可以使用其他设置（例如 automix 数据集）对其进行微调：

```bash
dora run -d -f 81de367c continue_from=81de367c dset=auto_mus variant=finetune
```

请注意，您需要同时使用 `-f 81de367c` 和 `continue_from=81de367c`。前者表示应使用 `81de367c` 的超参数作为配置的起点。
后者表示应使用 `81de367c` 的权重作为求解器的起点。

## 模型评估

您的模型将每 20 个 epoch 使用来自 MDX 的新 SDR 定义自动进行评估。
旧式 SDR（速度相当慢）只会在训练结束时进行。

## 模型导出

为了将您的模型用于其他命令（例如用于分离的 `demucs` 命令），您必须导出它。为此，请运行：

```bash
python3 -m tools.export 9357e12e [OTHER SIGS ...]  # 替换为适当的签名。
```

模型将存储在 `release_models/` 下。您可以通过以下标志将它们与 `demucs` 分离命令一起使用：
```bash
demucs --repo ./release_models -n 9357e12e my_track.mp3
```

### 模型集合包 (Bag of models)

如果您想组合多个模型，可能为每个源分配不同的权重，您可以将 `demucs/remote/mdx.yaml` 复制到 `./release_models/my_bag.yaml`。然后，您可以编辑模型列表（所有使用的模型都应该已首先导出）以及每个源和模型的权重（列表的列表，外层列表是模型，内层列表是源）。然后，您可以像这样使用您的模型集合包：

```bash
demucs --repo ./release_models -n my_bag my_track.mp3
```

## 模型评估

您可以使用以下命令评估任何预训练模型或模型集合包：
```bash
python3 -m tools.test_pretrained -n NAME_OF_MODEL [EXTRA ARGS]
```
其中 `NAME_OF_MODEL` 是模型集合包的名称（例如 `mdx`、`repro_mdx_a`），
或者是模型集合包中某个模型的单个 Dora 签名。您可以传递 `EXTRA ARGS` 来自定义测试选项，例如随机移位的次数（例如 `test.shifts=2`）。这将计算旧式 SDR，并且可能需要相当长的时间。

对于在本地训练的自定义模型，您需要使用 `--repo ./release_models` 标志来指明您希望使用本地模型仓库，例如：
```bash
python3 -m tools.test_pretrained --repo ./release_models -n my_bag
```

## 用于检索模型的 API

您可以使用以下 API 在 Python 中检索官方发布的模型：
```python
from demucs import pretrained
from demucs.apply import apply_model
bag = pretrained.get_model('htdemucs')    # 用于模型集合包或命名模型
                                          # （这只是一个包含1个模型的集合包）。
model = pretrained.get_model('955717e8')  # 使用单个模型的签名。

bag.models                       # 单个模型的列表
stems = apply_model(model, mix)  # 将模型应用于给定的混音。
```

## 模型动物园 (Model Zoo)

### 混合 Transformer Demucs

混合 Transformer 模型的配置可在以下位置找到：

```shell
dora grid mmi --dry_run --init
dora grid mmi_ft --dry_run --init  # 在每个源上进行微调。
```

我们特别发布了 `955717e8`，这是一个使用 5 个层、512 个通道、10 秒训练段长度的混合 Transformer Demucs。我们还发布了其微调版本，每个源都有一个模型：`f7e0c4bc`、`d12395a8`、`92cfc3b6`、`04573f0d`（分别为鼓、贝斯、其他、人声）。
模型 `955717e8` 也被命名为 `htdemucs`，而模型集合包则作为 `htdemucs_ft` 提供。

我们还发布了 `75fc33f5`，这是一个在相同数据集上训练的常规混合 Demucs，
可用作 `hdemucs_mmi`。

### 来自 MDX 2021 竞赛的模型

这里简要介绍了用于 MDX 提交的模型，无论是 A 赛道（仅限 MusDB HQ）还是 B 赛道（允许额外的训练数据）。训练分为两个阶段，第二阶段是在 automix 生成的数据集上进行微调。
所有微调模型都可在我们的 AWS 仓库中找到（您可以使用 `demucs.pretrained.get_model(SIG)` 检索它）。模型集合包可通过执行 `demucs.pretrained.get_model(NAME)` 获得，其中 `NAME` 为 `mdx`（用于 A 赛道）或 `mdx_extra`（用于 B 赛道）。

#### A 赛道

4 个模型是：

- `0d19c1c6`: 从 `9357e12e` 在 automix 数据集上微调而来
- `7ecf8ec1`: 从 `e312f349` 在 automix 数据集上微调而来
- `c511e2ab`: 从 `81de367c` 在 automix 数据集上微调而来
- `7d865c68`: 从 `80a68df8` 在 automix 数据集上微调而来

4 个初始模型（微调前）是：

- `9357e12e`: 64 通道纯时域改进版 Demucs，具有新的残差分支、组归一化和奇异值惩罚。
- `e312f349`: 64 通道纯时域改进版，具有新的残差分支、组归一化和奇异值惩罚，使用仅关注鼓和贝斯的损失函数进行训练。
- `81de367c`: 48 通道混合模型，具有残差分支、组归一化、奇异值惩罚和幅度谱图。
- `80a68df8`: 与 `b5559babb` 相同，但使用 CaC 和不同的随机种子，以及在最外层对不同频段使用不同的权重。

混合模型对除贝斯外的所有源使用相等权重进行组合。
`0d19c1c6`（时域）用于鼓和贝斯。`7ecf8ec1` 仅用于贝斯。

您可以使用以下命令一次性查看所有超参数（所有通用超参数共用一行，然后只显示不同的超参数），以及用于 `mdx_q` 模型的 DiffQ 变体：
```
dora grid mdx --dry_run --init
dora grid mdx --dry_run --init
```

#### B 赛道

- `e51eebcc`
- `a1d90b5c`
- `5d2d6c55`
- `cfa93e08`

所有模型都是具有不同随机种子的 48 通道混合 Demucs。其中两个使用 CaC，两个使用带掩码的幅度谱图。
所有模型对所有源使用相等权重进行组合。

B 赛道的情况有点混乱，在不同的数据集上进行了大量的微调。我不会在这里描述完整的模型谱系，
但所有信息都可以通过 `dora info -f SIG` 命令访问。

同样，您可以执行（这些将包含一些额外的行，用于在没有 MusDB 测试集作为训练数据的情况下进行训练，以及额外的 DiffQ XP）：
```
dora grid mdx_extra --dry_run --init
```

### 可复现性与消融研究

我更新了论文，以报告在一个比竞赛所用设置更同质化的设置下的数值。
在 MusDB HQ 上，我仍然需要结合使用纯时域模型和混合模型才能达到最佳性能。
实验在网格文件 [repro.py](../demucs/grids/repro.py) 和用于在真实混音数据集上进行微调的 [repro_ft._py](../demucs/grids/repro_ft.py) 中提供。

新的模型集合包达到了 7.64 的 SDR（而原始 A 赛道模型为 7.68）。它使用了2个使用残差分支、局部注意力和 SVD 惩罚训练的纯时域模型，
以及2个具有相同特性并使用 CaC 表示的混合模型。
我们对所有模型在所有源上的性能进行等权重平均，这与原始 A 赛道模型的做法不同。我们训练了 600 个 epoch，而之前是 360 个。

新的模型集合包作为预训练模型的一部分提供，名称为 `repro_mdx_a`。
纯时域模型集合包命名为 `repro_mdx_a_time_only`，纯混合模型集合包命名为 `repro_mdx_a_hybrid_only`。
有关训练的更多信息，请查阅论文。

[dora]: https://github.com/facebookresearch/dora