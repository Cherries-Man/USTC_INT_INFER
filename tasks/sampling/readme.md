## 任务要求

你需要完成以下几种采样方法

1. top-k
2. greedy
3. temperature
4. top-p
5. beam search(bs = 1)

你需要在一周内完成该任务.

## 额外要求

1. beam search(bs > 1). 除了固有的困难以外, 需要考虑一个边界条件: 如果两个beam, 一个遇到了EOS, 另一个没有, 怎么办?
2. 使用Sampler来设计整个采样流程. 有的时候, 可以同时使用两种采样方案(top-p-k), 而有的时候采样方案之间是不兼容的(greedy vs others). 能否设计出一种架构, 完成这些要求? 同时保持对新的采样方法开放. (你正在逐渐重新造出transformers......但是更好的软件工程:D )
3. length penalty
4. 测量各种sampling的时间开销(注意torch.cuda.synchronize). 如果是beam search, 还可以检查一下beam scores的内存开销. 