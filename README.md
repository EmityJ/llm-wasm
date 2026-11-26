# llm-wasm

用手写 WAT 把 **Qwen3-0.6B** 的前向计算移植到 WebAssembly，在浏览器或 Node.js 上使用 CPU 运行。单文件网页内嵌 Wasm、脚本和样式，支持离线多轮对话、流式回复和停止生成。

> 本项目基于 Alexey Fateev（alesha-pro）的 [llm.asm](https://github.com/alesha-pro/llm.asm)，上游基线为 [48feccb](https://github.com/alesha-pro/llm.asm/commit/48feccbfd57cb669440f31ff64958d95bb8b848e)。原始文档见 [README.upstream.md](README.upstream.md)，代码沿用原项目的 [MIT 许可](LICENSE)。

## 在浏览器里对话

1. 在 [chat.html 文件页](dist/chat.html)下载原始文件，再用支持 WebAssembly SIMD128 的浏览器打开。
2. 点击「选择权重文件」，选中本机的 `qwen3-0.6b-q8_0.asmbin`。已有原项目导出的 v1 或 v2 格式文件时可直接使用。
3. 等待「模型已就绪」，输入消息并按 Enter。Shift + Enter 换行，「新对话」清空当前会话的上下文。

从磁盘打开时，页面直接运行推理。通过 HTTP 打开时会优先使用 Worker，浏览器限制 Worker 启动时自动切换到页内推理。模型文件和对话由浏览器在本机处理。

Q8_0 v2 权重约 **614 MiB**，默认 512 词元的 KV 缓存约 **112 MiB**。推理使用单线程 CPU 和贪心解码。

## 本地准备权重

安装 NumPy 和 Hugging Face CLI，在仓库根目录下载模型并转换。模型由 Qwen 团队发布，按 Apache 2.0 许可提供。详见 [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)。

```sh
python -m pip install numpy huggingface_hub
hf download Qwen/Qwen3-0.6B config.json model.safetensors tokenizer.json --local-dir models/qwen3-0.6b-bf16
python scripts/export_weights.py --model-dir models/qwen3-0.6b-bf16 --out models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16
```

## 从源码构建

使用 Node.js 20 或更新版本、GNU Make 和 WABT 的 `wat2wasm`。

```sh
make wasm-chat
```

Windows 上可用 npm 调用 WABT。

```powershell
New-Item -ItemType Directory -Force build | Out-Null
npx --yes --package=wabt@1.0.39 wat2wasm 04-wasm-inference/qwen3.wat -o build/qwen3.wasm
node 04-wasm-inference/build-chat.mjs
```

构建产物是 `build/chat.html` 和 `build/qwen3.wasm`。仓库的 [dist/](dist/) 提供对应的预编译文件。

### Node.js 命令行

生成 `build/qwen3.wasm` 后，从仓库根目录运行。

```sh
node 04-wasm-inference/run.mjs -m models/qwen3-0.6b-q8_0.asmbin -p "用一句话解释 CPU 缓存。" -n 32
```

浏览器 API 和命令行参数见 [Wasm 使用说明](04-wasm-inference/README.md)。

## 源码与验证

- [WAT 推理核心](04-wasm-inference/qwen3.wat)用 SIMD128 完成 Q8_0 矩阵运算，包含 RMSNorm、QK-Norm、RoPE、GQA 和 SwiGLU。
- [JavaScript 宿主](04-wasm-inference/qwen3.mjs)流式加载原格式权重，使用模型内的 BPE 词表，复用匹配的 KV 缓存。
- [原始 NASM 实现](README.upstream.md)包含 Linux 推理、裸机推理和 Mini-GPT 训练。

用真实 Qwen3-0.6B 权重验证过 Node.js 与浏览器，多轮回复、停止生成和新对话均已测试。ASMBIN v1、v2 两个用例生成的 24 个词元与仓库的 [NumPy 参考实现](scripts/verify_ref.py)一致。
