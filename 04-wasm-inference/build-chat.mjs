import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

const [template, core, runtime, wasm, license] = await Promise.all([
  readFile(new URL("./chat.template.html", import.meta.url), "utf8"),
  readFile(new URL("./qwen3.mjs", import.meta.url), "utf8"),
  readFile(new URL("./chat-runtime.mjs", import.meta.url), "utf8"),
  readFile(new URL("../build/qwen3.wasm", import.meta.url)),
  readFile(new URL("../LICENSE", import.meta.url), "utf8"),
]);

const placeholder = "/*__CHAT_RUNTIME__*/";
const noticePlaceholder = "<!--__PROJECT_NOTICE__-->";
if (!template.includes(placeholder) || !template.includes(noticePlaceholder)) {
  throw new Error("HTML 模板缺少推理脚本或许可占位符");
}
if (!WebAssembly.validate(wasm)) throw new Error("无法嵌入无效的 qwen3.wasm");

// 同一份普通脚本可在页面和 Worker 中运行，推理过程共用一个实现。
const browserCore = core.replace(/^export (?=async function loadQwen3\()/m, "");
const payload = `const connectChatRuntime = (() => {
  "use strict";
${browserCore}
${runtime}
  const wasm = Uint8Array.from(atob("${wasm.toString("base64")}"), (c) => c.charCodeAt(0));
  return (emit) => createChatRuntime(loadQwen3, CHAT_TURN_END, wasm, emit);
})();

if (typeof document === "undefined") {
  const receive = connectChatRuntime((data) => self.postMessage(data));
  self.addEventListener("message", ({ data }) => receive(data));
}`;
if (/<\/script/i.test(payload)) throw new Error("推理脚本包含 HTML 结束标签");
new Script(payload, { filename: "chat-runtime.js" });

const path = new URL("../build/chat.html", import.meta.url);
const notice = "<!--\n基于 https://github.com/alesha-pro/llm.asm\n\n" + license.trim() + "\n-->";
const html = template.replace(noticePlaceholder, () => notice)
  .replace(placeholder, () => payload)
  .replaceAll("\r\n", "\n");
await writeFile(path, html);
console.log("已生成 " + fileURLToPath(path));
