function createChatRuntime(loadModel, endTurn, wasm, emit) {
  let model;
  let history = [];
  let generating = false;
  let stopped = false;

  async function load(file) {
    if (file.size === 0) throw new Error("模型文件为空");
    let read = 0;
    let lastPercent = 0;
    const stream = file.stream().pipeThrough(new TransformStream({
      transform(chunk, controller) {
        read += chunk.byteLength;
        const percent = Math.floor(read / file.size * 100);
        if (percent >= lastPercent + 2 || percent === 100) {
          lastPercent = percent;
          emit({ type: "progress", percent });
        }
        controller.enqueue(chunk);
      },
    }));
    model = await loadModel(wasm, stream);
    emit({ type: "ready", capacity: model.contextLength });
  }

  async function reply(prompt, maxTokens) {
    generating = true;
    stopped = false;
    try {
      const ids = [...history, ...model.encode(prompt)];
      const room = model.contextLength - ids.length - endTurn.length;
      if (room < 1) {
        throw new RangeError(history.length
          ? "这条消息放不进当前上下文。可以缩短消息或开启新对话。"
          : "消息超过上下文上限，请缩短后发送。");
      }
      const limit = Math.min(maxTokens, room);
      const generated = [];
      for (const step of model.generateSteps(ids, limit)) {
        if (step.prefill !== undefined) {
          if (step.prefill % 4 === 0 || step.prefill === ids.length) {
            emit({ type: "prefill", position: step.prefill, total: ids.length });
          }
        } else {
          if (step.tokenId !== undefined) generated.push(step.tokenId);
          emit({ type: "token", text: step.text, used: ids.length + generated.length });
        }
        // 每次前向计算后让出执行权，以便刷新页面并接收停止请求。
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (stopped) break;
      }
      const accepted = !stopped || generated.length > 0;
      if (accepted) history = [...ids, ...generated, ...endTurn];
      const reason = stopped ? "stopped" : generated.length < limit ? "eos" : limit === room ? "context" : "limit";
      emit({ type: "done", accepted, reason, used: history.length });
    } catch (error) {
      emit({ type: "reply-error", message: error.message, used: history.length });
    } finally {
      generating = false;
    }
  }

  return function receive(data) {
    switch (data.type) {
      case "load":
        load(data.file).catch((error) => {
          const unsupported = typeof WebAssembly === "undefined" || error instanceof WebAssembly.CompileError;
          const message = unsupported
            ? "当前页面无法运行内嵌的 WebAssembly。请在支持 Wasm SIMD 的浏览器中打开文件。"
            : error.message;
          emit({ type: "load-error", message });
        });
        break;
      case "reply":
        if (model && !generating) void reply(data.prompt, data.maxTokens);
        break;
      case "stop":
        if (generating) stopped = true;
        break;
      case "reset":
        if (!generating) history = [];
        break;
    }
  };
}
