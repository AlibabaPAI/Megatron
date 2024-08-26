# Megatron LLaMA3-8B与LLaMA3-70B模型测试流程

## 环境与数据集准备

- DLC镜像版本：nvcr.io/nvidia/pytorch:24.06-py3
- LLaMA3仓库：
    ```bash
    git clone https://github.com/meta-llama/llama3.git
    ```
- 安装依赖：
    ```bash
    pip install tiktoken flash-attn modelscope nltk -i https://mirrors.aliyun.com/pypi/simple/

    cd llama3
    pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
    ```

- 预处理数据，在`Megatron-LM`目录下运行，处理好的数据将保存在`data`目录下。
    ```bash
    bash examples/preprocess_data.sh
    ```

## 运行测试
参考`examples/llama3-8b.sh`与`examples/llama3-70b.sh`脚本，设置DLC最终的启动命令为其中某一行即可执行特定配置下的测试，例如：
```bash
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 8 --attn-type flash --seq-len 8192 --tp 2 --gc --gc-cnt 19
```
上述命令中的参数含义如下：
- `--random-init`：随机初始化
- `--mbs 2`：Micro batch size为2
- `--gbs 8`：Global batch size为8
- `--attn-type flash`：使用FlashAttention，可选项为`flash`、`fused`和`unfused`，分别对应FlashAttention、FusedAttention和UnfusedAttention
- `--seq-len 8192`：序列长度为8192
- `--tp 2`：Tensor parallel degree为2
- `--gc`：开启GC
- `--gc-cnt 19`：GC层数为19

如需更多定制化配置，请根据需求修改脚本中的参数设置。

