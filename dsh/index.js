/**
 * mvplus-subtitle as a DeepSeek Harness plugin.
 *
 * The pipeline itself is Python (GPU ASR, forced alignment, paid translation).
 * This plugin is deliberately thin: it never reimplements pipeline logic, it
 * runs `mvsub <subcommand>` and reports the result. Keeping one implementation
 * means the CLI, the docs and the tools cannot drift apart.
 *
 * Every tool is read-only unless its name says otherwise (`mvsub_run`,
 * `mvsub_translate`). `mvsub_run` downloads, spends money on translation and
 * uploads back to the cloud drive, so its description tells the model to
 * confirm scope with the user first.
 *
 * @module dsh-plugin-mvplus-subtitle
 */
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { defineTool } from "@deepseek-ai/dsh-tools";

const name = "mvplus-subtitle";
const inject = ["tools"];

const OUTPUT_TAIL = 12000;

/**
 * Per-user plugin settings, read on every call so changing them needs no restart.
 *
 * The host process environment is fixed at boot, so an env-var-only contract
 * would force a DSH restart whenever the interpreter path changes — and a bare
 * `python` on PATH fails with an opaque "No module named mvsub".
 */
function pluginConfigPath() {
	return join(homedir(), ".config", "mvsub", "plugin.json");
}

function pluginConfig() {
	try {
		// Windows editors (Notepad, PowerShell 5.1) happily write a UTF-8 BOM,
		// which JSON.parse rejects; strip it instead of failing the whole config.
		const text = readFileSync(pluginConfigPath(), "utf8").replace(/^\uFEFF/, "");
		const parsed = JSON.parse(text);
		return parsed && typeof parsed === "object" ? parsed : {};
	} catch {
		return {};
	}
}

/** Interpreter that has the `mvsub` package installed. */
function pythonPath() {
	const configured = process.env.MVSUB_PYTHON || pluginConfig().python;
	if (configured) return configured;
	return process.platform === "win32" ? "python.exe" : "python";
}

/** Working directory for the CLI (the repo checkout, usually). */
function workingDirectory() {
	return process.env.MVSUB_CWD || pluginConfig().cwd || process.cwd();
}

function tail(text, limit = OUTPUT_TAIL) {
	if (text.length <= limit) return text;
	return `…（前 ${text.length - limit} 字已省略）\n${text.slice(-limit)}`;
}

/**
 * Run `python -m mvsub <args>` and collect its output.
 * @param args - CLI arguments after the subcommand.
 * @param options - timeout in ms, or an explicit cwd.
 * @returns exit code plus captured streams.
 */
function runCli(args, { timeoutMs = 120000, cwd = workingDirectory() } = {}) {
	return new Promise((resolve) => {
		let child;
		const settings = pluginConfig();
		const extraEnv =
			settings.env && typeof settings.env === "object" ? settings.env : {};
		try {
			child = spawn(pythonPath(), ["-m", "mvsub", ...args], {
				cwd,
				stdio: ["ignore", "pipe", "pipe"],
				env: {
					...process.env,
					...extraEnv,
					PYTHONIOENCODING: "utf-8",
					PYTHONUTF8: "1",
				},
				windowsHide: true,
			});
		} catch (error) {
			resolve({ code: -1, stdout: "", stderr: `无法启动解释器 ${pythonPath()}: ${error.message}` });
			return;
		}
		let stdout = "";
		let stderr = "";
		let settled = false;
		const finish = (result) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve(result);
		};
		const timer = setTimeout(() => {
			try {
				child.kill();
			} catch {
				/* already gone */
			}
			finish({
				code: -2,
				stdout: tail(stdout),
				stderr: `超时（${Math.round(timeoutMs / 1000)} 秒）已终止。长批次请用 mvsub_status 查看，或用终端跑。`,
			});
		}, timeoutMs);
		child.stdout?.on("data", (chunk) => {
			stdout += chunk.toString("utf8");
		});
		child.stderr?.on("data", (chunk) => {
			stderr += chunk.toString("utf8");
		});
		child.on("error", (error) => {
			finish({ code: -1, stdout: tail(stdout), stderr: `进程错误: ${error.message}` });
		});
		child.on("close", (code) => {
			finish({ code: code ?? -1, stdout: tail(stdout), stderr: tail(stderr) });
		});
	});
}

/** Shared result shape for every tool. */
const resultSchema = {
	type: "object",
	additionalProperties: false,
	properties: {
		ok: { type: "boolean", required: true, description: "退出码是否为 0" },
		exitCode: { type: "integer", required: true },
		output: { type: "string", required: true, description: "合并后的 stdout/stderr（尾部截断）" },
	},
};

function renderResult(_args, value) {
	return [{ type: "text", text: value.ok ? value.output || "完成（无输出）" : `失败（退出码 ${value.exitCode}）\n${value.output}` }];
}

/**
 * Wrap one CLI invocation as a tool result.
 * @param args - CLI arguments after the subcommand.
 * @param options - timeout and cwd overrides.
 * @returns the tool's structured result.
 */
async function invoke(args, options) {
	const { code, stdout, stderr } = await runCli(args, options);
	let output = [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
	// The single most likely first failure: the configured interpreter has no mvsub.
	if (code !== 0 && /No module named ['"]?mvsub/.test(output)) {
		output +=
			`\n\n此解释器里没有 mvsub。把 ${pluginConfigPath()} 写成 ` +
			`{"python": "<装了 mvsub 的 python 绝对路径>", "cwd": "<仓库目录>"} 后重试；` +
			"该文件每次调用都会重新读取，不需要重启 DSH。";
	}
	return { ok: code === 0, exitCode: code, output };
}

const cidParameter = {
	remoteCid: {
		type: "string",
		required: true,
		description: "115 目录的 cid（目录 id）。可用 mvsub cloud ls --remote-dir <路径> 查。",
	},
};

/**
 * Register the mvsub tools on the harness tool registry.
 * @param ctx - registrant context carrying the tool registry.
 */
function apply(ctx) {
	ctx.tools.register(defineTool({
		name: "mvsub_doctor",
		description:
			"检查 mvplus-subtitle 的运行环境：Python 版本、ffmpeg/ffprobe、115 cookie、翻译密钥、模型目录、可写目录。" +
			"只读、不联网。第一次使用或流水线报错时先跑这个，它会把缺什么直接列出来。",
		parameters: {},
		output: { schema: resultSchema, render: renderResult },
		async execute() {
			return invoke(["doctor"]);
		},
		presentCall: () => ({ card: "generic", title: "体检 mvplus-subtitle", kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_config",
		description: "打印 mvplus-subtitle 解析后的全部路径（模型、归档、下载、工作目录）以及生效的配置文件。用于排查「配了但没生效」。",
		parameters: {},
		output: { schema: resultSchema, render: renderResult },
		async execute() {
			return invoke(["config"]);
		},
		presentCall: () => ({ card: "generic", title: "查看 mvsub 配置", kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_scan",
		description:
			"只读预检：列出 115 某个目录里**缺中文字幕**的视频，不下载、不上传、不花钱。" +
			"正式跑之前先用它确认要处理哪些片。",
		parameters: {
			...cidParameter,
			limit: {
				type: "integer",
				description: "只看前 N 部（0 表示全部）。",
			},
		},
		output: { schema: resultSchema, render: renderResult },
		async execute(args) {
			const cli = ["scan", "--remote-cid", args.remoteCid];
			return invoke(cli);
		},
		presentCall: (args) => ({ card: "generic", title: `扫描目录 ${args.remoteCid}`, kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_status",
		description: "查看某个 115 目录的补字幕批次进度：每部片到了哪一段、哪些失败、哪些已完成。只读。",
		parameters: { ...cidParameter },
		output: { schema: resultSchema, render: renderResult },
		async execute(args) {
			return invoke(["status", "--remote-cid", args.remoteCid]);
		},
		presentCall: (args) => ({ card: "generic", title: `查询进度 ${args.remoteCid}`, kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_run",
		description:
			"正式跑整条流水线：下载 → 本地识别 → 强制对齐 → 翻译 → 校验 → 回传云盘。" +
			"**会下载几十 GB、会花翻译费、会往云端写文件**，因此先和用户确认范围，建议先 `limit: 1` 试跑一部。" +
			"付费翻译必须绑定显式预算台账：`budgetFile` 与 `budgetCny` 都要给，缺一个直接失败（这是刻意的闸门）。" +
			"中断后用同样参数重跑即续跑，已完成的部分不会重算、不会重复付费。",
		parameters: {
			...cidParameter,
			budgetFile: {
				type: "string",
				required: true,
				description: "共享预算台账文件路径；续跑必须用同一个文件和同一个上限。",
			},
			budgetCny: {
				type: "number",
				required: true,
				description: "本轮翻译预算上限（人民币，含重试）。",
			},
			only: {
				type: "array",
				items: { type: "string" },
				description: "只跑文件名含这些关键词的片（可多个）。试跑验收时用。",
			},
			limit: { type: "integer", description: "最多跑 N 部（0 或省略表示全部）。" },
			timeoutSeconds: { type: "integer", description: "等待上限（秒），默认 600。整批很久时先只看 status。" },
		},
		output: { schema: resultSchema, render: resultRender },
		async execute(args) {
			const cli = ["run", "--remote-cid", args.remoteCid];
			for (const keyword of args.only ?? []) cli.push("--only", keyword);
			if (args.limit) cli.push("--limit", String(args.limit));
			cli.push("--budget-file", args.budgetFile, "--budget-cny", String(args.budgetCny));
			return invoke(cli, { timeoutMs: Math.max(30, args.timeoutSeconds ?? 600) * 1000 });
		},
		presentCall: (args) => ({ card: "generic", title: `跑补字幕 ${args.remoteCid}`, kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_verify",
		description:
			"只读复核一对字幕：可解析、cue 数与时间轴一致、中文里没有假名残留、CPS 与时长在阈值内。" +
			"本地运行、不联网、不写状态。退出码 0 表示硬判据全过；它**不**证明语义翻译正确。",
		parameters: {
			video: { type: "string", required: true, description: "视频路径；没有视频时传 - 。" },
			sourceSrt: { type: "string", required: true, description: "原文 SRT（如 movie.ja.srt）。" },
			targetSrt: { type: "string", required: true, description: "中文 SRT。" },
		},
		output: { schema: resultSchema, render: renderResult },
		async execute(args) {
			return invoke(["verify", args.video, args.sourceSrt, args.targetSrt]);
		},
		presentCall: () => ({ card: "generic", title: "复核字幕", kind: "other" }),
	}));

	ctx.tools.register(defineTool({
		name: "mvsub_translate",
		description:
			"只翻译一份本地原文 SRT，不上传、不下载。默认走 DeepSeek 付费接口并限制在优惠时段；" +
			"付费请求必须绑定显式预算台账（`budgetFile` + `budgetCny` 都要给），这是防止误花钱的闸门。" +
			"已有译文会按 checkpoint 复用，不会重复付费。",
		parameters: {
			srt: { type: "string", required: true, description: "原文 SRT 路径。" },
			budgetFile: {
				type: "string",
				required: true,
				description: "共享预算台账文件路径。",
			},
			budgetCny: {
				type: "number",
				required: true,
				description: "本轮预算上限（人民币）。",
			},
			output: { type: "string", description: "输出路径，默认与原文同目录同名（不带语言后缀）。" },
			model: { type: "string", description: "模型名，默认 deepseek-flash。" },
			offPeakOnly: { type: "boolean", description: "只在优惠时段请求，默认 true。" },
			timeoutSeconds: { type: "integer", description: "等待上限（秒），默认 900。" },
		},
		output: { schema: resultSchema, render: renderResult },
		async execute(args) {
			const cli = [
				"translate", args.srt,
				"--grounded",
				"--batch-size", "16",
				"--max-workers", "2",
				"--budget-file", args.budgetFile,
				"--budget-cny", String(args.budgetCny),
			];
			if (args.output) cli.push("--output", args.output);
			if (args.model) cli.push("--model", args.model);
			// `--off-peak-only` is a store_true flag, so "default on" means passing it unless disabled.
			if (args.offPeakOnly !== false) cli.push("--off-peak-only");
			return invoke(cli, { timeoutMs: Math.max(30, args.timeoutSeconds ?? 900) * 1000 });
		},
		presentCall: (args) => ({ card: "generic", title: `翻译 ${args.srt}`, kind: "other" }),
	}));
}

/** Batch runs deserve a longer render: they report per-file outcomes. */
function resultRender(_args, value) {
	const head = value.ok ? "流水线已跑完" : `流水线报错（退出码 ${value.exitCode}）`;
	return [{ type: "text", text: `${head}\n${value.output}` }];
}

export { apply, inject, name };
