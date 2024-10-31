## 任务要求

你需要完成`llama.py`中的`Engine`类. 并通过`test.py`的检查. 你只需要实现batchsize=1的版本即可.

你必须使用Llama-3-8B-Instruct的权重(通过设置环境变量HF_MODELS_CACHE). 你只能使用torch/transformers/numpy和内置库.


torch: 你不能使用和Attention有关的任何实现. 例如MultiAttention, 或者SDPA. 除此之外都可以使用

transformers: 你仅能继承LlamaPretrainedModel, 使用其from_pretrained方法加载权重. 

你不能使用其他任何类和函数.

你需要在2周内完成该任务.

参考资料:

modeling_llama.py

vllm的Llama.py

## 额外要求

1. 满足Ruff规范.
2. 所有参数标注类型. type checking无错误.
3. 类型命名使用双驼峰.
4. 方法命名使用全小写下划线.
5. modeling_llama中有很多冗余代码. 如果你完全复制了它, 最好能删节至最简状态.
6. 分别报告prefill和decoding的时间. 
7. 观察内存用量曲线; torch原生实现的llama中, 单层内存用量最大的tensor是什么?
8. HF提供了LlamaTokenizer类进行分词. 但如果将test.py中的AutoTokenizer替换成LlamaTokenizer就会报错. 为什么? Llama, Llama2, Llama3的分词器有何差异?