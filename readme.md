## 这是什么？

这是[USTC-INT](https://int-ustc.github.io/) 的大模型推理入门练习仓库。我们会在这里（main分支）发布任务，您需要向自己的分支提交代码。

任务要求是**必须**完成的内容; 额外要求是可选项. 

对于每个任务, 使用越少的代码越好. 任务本身是被设计成"只完成最基础功能即可"形式的, 每个任务之间互有关联.

**对于所有任务, 你都需要对于改动的部分给出它的相应伪代码**

## Tasks

### 从零Llama

该任务要求您从头完成Llama模型的构建。

仅允许的依赖：torch, transformers（仅允许使用其from_pretrained加载函数）

检查`tasks/llamascratch`获取更多信息。

### 推理Batch

该任务要求您执行对一个Batch的推理.

在之前的代码的基础上, 修改其, 使其可以推理不同长度的句子.

检查`tasks/batchify`获取更多信息。

### Sampling

该任务要求您完成各种采样核.

检查`tasks/sampling`获取更多信息。

### Rope

该任务将Rope操作从Attention解耦, 允许更灵活的Cache策略.

检查`tasks/rope`获取更多信息。

## Adavanced Topics

本节是各种高级技术的一个小指引. 完成上面的任务之后您应当至少对于Llama系列了如指掌, 但是编写出来的代码通常既不高效也缺乏可维护性. 下面的内容将至少对其中一方面大有裨益.

### setuptools

使用pyproject.toml + setup.py将自己的代码转换成可以安装的库代码. 

一定要动手自己写一遍pyproject.toml

### Linter

Linter我只推荐[Ruff](https://github.com/astral-sh/ruff).

打开Type checking, 文件里面一条波浪线没有就行了!(通常可以帮你提前避免50%的错误)

### Git开发流程

[这个视频](https://www.youtube.com/watch?v=uj8hjLyEBmU)就够了.

更进一步可以试着自己review代码(上面的视频仅仅针对提交者).

### 软件工程

在大模型开发过程中, 实际上软工和高性能代码的重要性是对半开的. 在之前的任务中也经常会提问如何在避免硬编码的情况下, 保持一个符合开闭原则的设计. 保持良好的设计可以大幅缩短你的开发时间和debug时间.

在编写分布式代码的时候这点更为重要. 实际上真正"困难"的技术部分只占20%, 而工程部分却占据了问题的80%. Meta的训练报告中就可以窥见一斑.

### 单元测试

pytest:https://docs.pytest.org/en/stable/

减少代码出错是多方面共同出力的成果. 1. 良好的设计; 2. **多写类型注释**; 3. 多写单元测试.

尤其是在之后进行GPU编程的时候, 由于GPU的异步问题, debug会非常痛苦, 单元测试几乎就是这种情况下唯一能够保护你的最后一道防线.

推荐的单元测试的单元是

1. cuda/triton核
2. 类
3. 类方法

注意, 一定要测试的是*稳定接口*, 对于一些不公开接口,或者经常变动的接口, 推荐写一个小的测试脚本即可. 因为一旦发生变动, 测试代码也必须变动, 如果为此编写单元测试反而浪费时间. 编写单元测试时, 务必覆盖所有可能情况, 利用好pytest的组合系统减少编写的代码量.

### 高效Attention

[Flash Attention](https://github.com/Dao-AILab/flash-attention)

Paged Attention: 大名鼎鼎的vllm已经不用自家的核了; 看看远方的[flashinfer](https://github.com/flashinfer-ai/flashinfer)吧

Sparse Attention: 这方面最有名的是xFormers

quiz: 如何使用flash attention推理不等长句子?

### Pytorch Profiler

[profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 将是你在纯python世界的最后一个最常使用到的东西. 养成好习惯: 编写单元测试 + profiler测量.

### torch.compile

[一行代码](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)进入JIT的世界. 不过请注意, dynamic=False的代码通常容易多次触发重编译.

**必要任务**: 使用torch.compile + 静态cache加速你之前的llama代码. 看看能到多快!

### Pytorch Cuda Stream

https://pytorch.org/docs/stable/generated/torch.cuda.stream.html

[臭名昭著](https://github.com/pytorch/pytorch/issues/59692)的Torch Cuda Stream. 了解Cuda的异步执行机制非常重要, 但是Torch提供的Stream效用非常有限.

### Openai Triton

https://triton-lang.org/main/index.html

推荐把tutorial里面的所有核都自己动手写一遍. 

### Nvidia Cuda

> 我真的需要放链接吗 :X

### Cuda Graphs

该操作经常和torch.compile连用. 具体可以参考https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

值得一提的是, Cuda Graph和一些关键的加速技巧冲突, 例如Continous Batching.

