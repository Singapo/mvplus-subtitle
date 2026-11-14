/**
 * CLI argument builders for the mvplus-subtitle tools.
 *
 * Kept free of DeepSeek Harness imports so they can be unit-tested with plain
 * Node, without a loaded harness or a `node_modules` tree.
 *
 * @module mvplus-subtitle/args
 */

/**
 * The only endpoint the pipeline's budget mode accepts.
 *
 * `translate_srt_batch` defaults to a local Ollama endpoint and `qwen2.5:14b`,
 * and budget mode rejects both outright. Passing them explicitly is not
 * cosmetic: without `--url` every call fails parameter validation before a
 * subtitle is ever read.
 */
export const OFFICIAL_URL = "https://api.deepseek.com";

/** The only model the pipeline's budget mode accepts. */
export const OFFICIAL_MODEL = "deepseek-flash";

/**
 * Build the `translate` argv.
 * @param {object} args - tool arguments; budget fields are mandatory upstream.
 * @returns {string[]} CLI arguments after the program name.
 */
export function buildTranslateArgs(args) {
	const cli = [
		"translate",
		args.srt,
		"--grounded",
		"--url",
		args.url || OFFICIAL_URL,
		"--model",
		args.model || OFFICIAL_MODEL,
		"--batch-size",
		"16",
		"--max-workers",
		"2",
		"--budget-file",
		args.budgetFile,
		"--budget-cny",
		String(args.budgetCny),
	];
	if (args.output) cli.push("--output", args.output);
	// `--off-peak-only` is a store_true flag, so "default on" means passing it unless disabled.
	if (args.offPeakOnly !== false) cli.push("--off-peak-only");
	return cli;
}

/**
 * Build the `run` argv for the whole-pipeline tool.
 * @param {object} args - tool arguments from `mvsub_run`.
 * @returns {string[]} CLI arguments after the program name.
 */
export function buildRunArgs(args) {
	const cli = ["run", "--remote-cid", args.remoteCid];
	for (const keyword of args.only ?? []) cli.push("--only", keyword);
	if (args.limit) cli.push("--limit", String(args.limit));
	cli.push("--budget-file", args.budgetFile, "--budget-cny", String(args.budgetCny));
	return cli;
}
