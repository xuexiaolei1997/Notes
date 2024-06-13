# 从零开始构建LLM

[书籍链接](https://www.manning.com/books/build-a-large-language-model-from-scratch)

## 1. LLM是什么

大语言模型，主要结构如下：从raw data中进行预训练，得出基础模型（这一部分可以了解一下元学习的概念），这个基础模型所拥有的基础能力为文本补全、短时任务的推理能力。

在基础模型之上，可以导入自己标记的数据进行训练，这一部分可以成为微调（finetune），得到自己的LLM，可以用于分类，总结，翻译，个人助理等任务。

![1716275709784](image/从零开始构建LLM/1716275709784.png)

transformer结构

![1716275687724](image/从零开始构建LLM/1716275687724.png)

BERT与GPT区别：BERT更多的使用于文本填空，GPT则是预测下一个单词。

![1716275758151](image/从零开始构建LLM/1716275758151.png)

构建大模型步骤

![1716275818354](image/从零开始构建LLM/1716275818354.png)

---

## 2. 文本数据处理

### 2.1 词嵌入

词嵌入的根本目的是为了将非数值数据转换为向量，这样才能放入计算机进行运算。常见词嵌入的有Word2Vec。在GPT架构中，没有使用这一技术，GPT3的嵌入大小达到了12288维。其中，GPT将词嵌入作为训练模型，不断调整。也就是说，GPT将词嵌入这一部分也进行训练。

![1716433691383](image/从零开始构建LLM/1716433691383.png)

### 2.2 标记文本

首先分词，再将分词的结果用字典标记token id，token id在进行词嵌入。

![1716433945337](image/从零开始构建LLM/1716433945337.png)

在分词完成后，转换为token id需要对应的字典表。字典表，可以自己构建，通过给每个单词指定唯一id，完成之后即可完成token与id之间的互相转换。

![1716434053588](image/从零开始构建LLM/1716434053588.png)

![1716434299643](image/从零开始构建LLM/1716434299643.png)

### 2.3 特殊处理

在将文本转换为token id时，字典表的大小，覆盖全不全面，对于tokenizer是一个很严峻的考验。如果需要转换的字符不在字典表中，就需要特殊处理，另外对于不同句子之间，也需要分割符。

![1716455910828](image/从零开始构建LLM/1716455910828.png)

不同句子，不同文本之间的合并，可以使用定义的分割符进行连接

![1716455950157](image/从零开始构建LLM/1716455950157.png)

### 2.4 字节对编码（Byte pair encoding, BPE）

使用包：tiktoken

对于未知词，使用分词，然后进行编码，并根据频率进行合并。

![1716778401378](image/从零开始构建LLM/1716778401378.png)

### 2.5 使用滑窗进行数据采样

![1716778580547](image/从零开始构建LLM/1716778580547.png)

![1716778596288](image/从零开始构建LLM/1716778596288.png)

不同步长下的LLM采样

![1716778622405](image/从零开始构建LLM/1716778622405.png)

2.6 创建token嵌入

![1716778710812](image/从零开始构建LLM/1716778710812.png)

这一部分从初始化权重中根据token id进行选择。

![1716778796944](image/从零开始构建LLM/1716778796944.png)

### 2.6 编码单词位置

在上一节中，会出现如下问题，当token id一致时，从权重矩阵中选择的向量也一致。

![1716778847577](image/从零开始构建LLM/1716778847577.png)

为了解决这一问题，引入了位置编码，这样就可以保证每一个编码是独一无二的

![1716778935403](image/从零开始构建LLM/1716778935403.png)

最后，所有的数据处理流程如下：

![1716778990725](image/从零开始构建LLM/1716778990725.png)

## 3. 编码注意力机制

本节主要关注内容如下

![1716780308367](image/从零开始构建LLM/1716780308367.png)

![1716780422961](image/从零开始构建LLM/1716780422961.png)

### 3.1 长序列建模的问题

上下文丢失

### 3.2 使用注意力机制获取数据依赖关系

![1716877433399](image/从零开始构建LLM/1716877433399.png)

### 3.3 自注意中输入的不同部分

“自我”是指该机制通过关联单个输入序列中的不同位置来计算注意权重的能力。它评估和学习输入本身的各个部分之间的关系和依赖关系，比如句子中的单词或图像中的像素。

#### 3.3.1 一个简单的自我注意机制，没有训练权重

![1716879045635](image/从零开始构建LLM/1716879045635.png)

点积越大，表示相关性/相似度越高

![1716881565809](image/从零开始构建LLM/1716881565809.png)

权重归一化

![1716882217614](image/从零开始构建LLM/1716882217614.png)

![1716882746585](image/从零开始构建LLM/1716882746585.png)

#### 3.3.2 计算所有输入的注意力权重

![1716945187106](image/从零开始构建LLM/1716945187106.png)

![1716945198064](image/从零开始构建LLM/1716945198064.png)

### 3.4 使用训练权重实现自注意力

![1716946588444](image/从零开始构建LLM/1716946588444.png)

#### 3.4.1 逐步计算注意力权重

首先引入三个权重矩阵${W_q}、{W_k}、{W_v}$，用于将输入embedding。

![1716947702233](image/从零开始构建LLM/1716947702233.png)

输入与W矩阵相乘，得出注意力分数

W为权重参数，w为注意力权重

权重参数是定义网络连接的基本学习系数，而注意力权重是动态的、上下文特定的值。

![1716971035083](image/从零开始构建LLM/1716971035083.png)

注意力分数计算注意力权重（不是归一化，而是使用平方根来进行计算）

![1717036962949](image/从零开始构建LLM/1717036962949.png)

最后计算上下文向量

![1717039022685](image/从零开始构建LLM/1717039022685.png)

#### 3.4.2 实现一个紧凑的自注意力类

```python
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
  
    def forward(self, x):
        keys = torch.dot(x, self.W_key)
        values = torch.dot(x, self.W_value)
        queries = torch.dot(x, self.W_query)

        attn_scores = torch.dot(queries, keys.T)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = torch.dot(attn_weights, values)
        return context_vec


```

![1717397936262](image/从零开始构建LLM/1717397936262.png)

```python
import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
  
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = torch.dot(queries, keys.T)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = torch.dot(attn_weights, values)
        return context_vec

```

### 3.5 用因果关系的注意力来隐藏未来的词语

使用掩码机制，

![1717401849186](image/从零开始构建LLM/1717401849186.png)

#### 3.5.1 应用因果注意力掩码

![1717401901352](image/从零开始构建LLM/1717401901352.png)

![1717402483387](image/从零开始构建LLM/1717402483387.png)

#### 3.5.2 使用dropout掩盖额外的注意力权重

在transformer架构中，dropout通常用在两个地方：计算注意力分数之后或者应用注意力权重之前

![1717558177482](image/从零开始构建LLM/1717558177482.png)

需要注意的是，dropout时，会将原数值进行放大，这样能够保证注意力权重的平衡。

#### 3.5.3 实现一个紧凑的因果注意类

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.reigster_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = torch.dot(queries, keys.T)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.dot(attn_weights, values)
        return context_vec
```

reigster_buffer在pytorch中并不是必要的，但是使用之后有如下好处：缓存会跟着模型自动移动到设备中。

![1717566719143](image/从零开始构建LLM/1717566719143.png)

### 3.6 将多头注意力扩展到多头注意力

#### 3.6.1 叠加多个单头注意层

![1717567513041](image/从零开始构建LLM/1717567513041.png)

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout) for _ in range(num_heads)]
        )
  
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

![1717661757839](image/从零开始构建LLM/1717661757839.png)

#### 3.6.2 通过权重分割实现注意力机制

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
  
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
  
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
```

![1718186662363](image/从零开始构建LLM/1718186662363.png)

## 4. 实现GPT并生成文本

### 4.1 实现一个LLM结构

![1718247386638](image/从零开始构建LLM/1718247386638.png)

![1718258629553](image/从零开始构建LLM/1718258629553.png)

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = DummtLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
  
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.range(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
  
    def forward(self, x):
        return x

class DummtLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5) -> None:
        super().__init__()
  
    def forward(self, x):
        return x
```

![1718262314570](image/从零开始构建LLM/1718262314570.png)

### 4.2 layer归一化

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
  
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

layer归一化一般放在多头注意力前后和最终输出之前

![1718265836573](image/从零开始构建LLM/1718265836573.png)

### 4.3 使用GELU激活函数实现前馈网络

在神经网络中，使用最广泛的是ReLU函数，但是在LLM，除了ReLU外，还有两种显著的激活函数：GELU (Gaussian Error Linear Unit) 和 SwiGLU (Sigmoid-Weighted Linear Unit)。GELU和SwiGLU分别是更复杂和光滑的包含高斯单元和s型门控线性单位的激活函数。他们可以表现的更好。

$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right]\right)
$$

![1718272903120](image/从零开始构建LLM/1718272903120.png)

在此基础上实现一个前馈网络，这个前馈网络至关重要，主要解决非线性问题，并且可以探索更丰富的空间。

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
  
    def forward(self, x):
        return self.layer(x)
```

### 4.4 增加短连接

![1718273582034](image/从零开始构建LLM/1718273582034.png)
