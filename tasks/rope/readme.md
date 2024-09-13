## 任务要求

现在, 我们存在如下需求: 每次计算attention时, 先将key从cache中取出, 然后再统一施加旋转编码. 请修改之前的代码, 使得该代码可以兼容 储存旋转编码后的key/储存旋转编码前的key 的cache, 并且实现这样的cache.

**你不可以使用if/else硬编码来解决此问题.**

该任务没有测试脚本(因为无法端到端测试). 你需要保证对于不同的Cache, 之前的代码依然可以工作.

你需要在1周内完成该任务.

## 额外要求

1. 比较PreRope和PostRope的时延差异.
2. 增加一个参数position_ids. 该张量会指明句子中每个token的旋转编码位置. 有关信息参阅[SpecInfer](https://arxiv.org/pdf/2305.09781).
3. 试着实现一下SinkCache(StreamLLM)
4. 推导一遍RoPE的计算公式. Llama的实际RoPE代码和公式有什么区别?
5. Llama3.1和Llama3的旋转编码有什么区别? 尝试将[这篇文章](https://unsloth.ai/blog/llama3-1)的代码改写成高效的torch代码.
6. 试着不添加RoPE运行一次模型. 直观感受一下RoPE有多么重要.