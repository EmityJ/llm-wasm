# Qwen3 WebAssembly 推理

用手写 WAT 实现 Qwen3 的前向计算。矩阵乘法使用 WebAssembly SIMD128，权重沿用 Q8_0、激活使用 FP32。模块在单线程里执行 RMSNorm、GQA、QK-Norm、RoPE、SwiGLU 和 KV 缓存更新。JavaScript 负责加载权重、复用模型里的 BPE 词表和合并表、提供 `Math.exp`、`Math.sin`、`Math.cos`，以及执行贪心解码。

## 构建和运行

从仓库根目录运行。构建需要 WABT 的 `wat2wasm`，命令行示例需要 Node.js 20 或更新版本。

```sh
make wasm
node 04-wasm-inference/run.mjs -m models/qwen3-0.6b-q8_0.asmbin -p "用一句话解释 CPU 缓存。" -n 32
```

Windows 上可用 npm 调用 `wat2wasm`。

```powershell
New-Item -ItemType Directory -Force build | Out-Null
npx --yes --package=wabt@1.0.39 wat2wasm 04-wasm-inference/qwen3.wat -o build/qwen3.wasm
```

直接使用[原有导出流程](../01-qwen3-inference/README.md#2-export-weights-to-asmbin-q8_0)生成的 `.asmbin`。支持 v2 FP16 缩放（约 614 MiB）和 v1 FP32 缩放（约 649 MiB）。默认上下文为 512 个词元，Qwen3-0.6B 的 FP32 KV 缓存在此长度下约占 112 MiB，可用 `-c` 调整。默认使用 Qwen3 的非思考对话模板，`--think` 启用思考模板，`--raw` 直接编码输入。

## 离线对话页

运行 `make wasm-chat` 生成 `build/chat.html`。如果已按上文的 Windows 命令生成 `build/qwen3.wasm`，也可以从仓库根目录运行 `node 04-wasm-inference/build-chat.mjs`。

双击打开 `build/chat.html`，选择已导出的 `.asmbin` 文件，然后输入消息。页面内嵌样式、脚本和 Wasm。直接从磁盘打开时，浏览器在页内推理。通过 HTTP 打开时优先使用 Worker，浏览器限制 Worker 启动时自动切换到页内推理。两种方式都支持流式回复、停止生成和复用对话的 KV 缓存。默认上下文为 512 个词元，达到上限后可点击「新对话」。

## 浏览器接口

在支持 WebAssembly SIMD128 的浏览器中，将以下代码作为模块 Worker 执行。示例中的 URL 以仓库根目录为网站根目录。模型也可以来自本地选中的文件，把 `weights.body` 替换为 `file.stream()`。

```js
import { loadQwen3 } from "/04-wasm-inference/qwen3.mjs";

const [wasm, weights] = await Promise.all([
  fetch("/build/qwen3.wasm"),
  fetch("/models/qwen3-0.6b-q8_0.asmbin"),
]);
if (!wasm.ok || !weights.ok) throw new Error("无法读取 Wasm 或模型文件");

const model = await loadQwen3(await wasm.arrayBuffer(), weights.body, { contextLength: 512 });
const { text } = model.generate(model.encode("你好，请介绍一下你自己。"), 32);
postMessage(text);
```

加载器逐块把权重写入 Wasm 线性内存。生成是同步操作，在 Worker 中运行可避免阻塞页面。模型内存还需容纳缓存和工作缓冲区，设备可用内存会影响能否加载。
