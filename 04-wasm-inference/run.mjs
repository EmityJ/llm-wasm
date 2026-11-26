import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { loadQwen3 } from "./qwen3.mjs";

try {
  const { values } = parseArgs({
    options: {
      model: { type: "string", short: "m" },
      prompt: { type: "string", short: "p" },
      "max-tokens": { type: "string", short: "n", default: "32" },
      context: { type: "string", short: "c" },
      raw: { type: "boolean" },
      think: { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });
  if (values.help) {
    console.log("用法  node 04-wasm-inference/run.mjs -m 模型.asmbin -p 提示词 [-n 32] [-c 512] [--raw | --think]");
  } else {
    if (!values.model || values.prompt === undefined) throw new Error("请用 -m 指定模型，用 -p 指定提示词");
    const wasm = await readFile(new URL("../build/qwen3.wasm", import.meta.url));
    const model = await loadQwen3(wasm, createReadStream(values.model), {
      contextLength: values.context === undefined ? undefined : Number(values.context),
    });
    const ids = model.encode(values.prompt, { raw: values.raw, think: values.think });
    const result = model.generate(ids, Number(values["max-tokens"]));
    process.stdout.write(result.text + "\n");
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
