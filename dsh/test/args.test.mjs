/**
 * Unit tests for the argv builders. Plain Node, no harness, no node_modules:
 * run `node dsh/test/args.test.mjs`.
 */
import assert from "node:assert/strict";
import { buildRunArgs, buildTranslateArgs, OFFICIAL_MODEL, OFFICIAL_URL } from "../args.js";

const base = { srt: "/media/movie.ja.srt", budgetFile: "/media/ledger.json", budgetCny: 12 };

// The regression this file exists for: without --url the CLI falls back to
// Ollama's localhost and budget mode rejects the call before reading anything.
const translateDefaults = buildTranslateArgs(base);
assert.ok(
	translateDefaults.includes("--url") && translateDefaults.includes(OFFICIAL_URL),
	"translate must pin the official endpoint",
);
assert.equal(
	translateDefaults[translateDefaults.indexOf("--url") + 1],
	OFFICIAL_URL,
);
assert.equal(
	translateDefaults[translateDefaults.indexOf("--model") + 1],
	OFFICIAL_MODEL,
);
assert.equal(
	translateDefaults[translateDefaults.indexOf("--budget-cny") + 1],
	"12",
);
assert.ok(translateDefaults.includes("--grounded"));
assert.ok(translateDefaults.includes("--off-peak-only"), "off-peak defaults to on");

// An explicit model must win over the default but still carry the endpoint.
const explicit = buildTranslateArgs({ ...base, model: "deepseek-reasoner" });
assert.equal(explicit[explicit.indexOf("--model") + 1], "deepseek-reasoner");
assert.equal(explicit[explicit.indexOf("--url") + 1], OFFICIAL_URL);

// Opting out of off-peak must remove the flag rather than emit an opposite flag.
const noGate = buildTranslateArgs({ ...base, offPeakOnly: false });
assert.ok(!noGate.includes("--off-peak-only"));
assert.ok(!noGate.some((a) => a.includes("no-off-peak")), "no such flag exists upstream");

// Optional output is only present when requested.
assert.ok(!translateDefaults.includes("--output"));
const withOutput = buildTranslateArgs({ ...base, output: "/media/movie.srt" });
assert.equal(withOutput[withOutput.indexOf("--output") + 1], "/media/movie.srt");

// run: repeated --only (the upstream flag is repeatable) and a numeric --limit.
const run = buildRunArgs({ remoteCid: "123", only: ["A", "B"], limit: 2, budgetFile: "l.json", budgetCny: 5 });
assert.deepEqual(run.slice(0, 3), ["run", "--remote-cid", "123"]);
assert.equal(run.filter((a) => a === "--only").length, 2, "--only is repeatable, one per keyword");
assert.equal(run[run.indexOf("--only") + 1], "A");
assert.equal(run[run.lastIndexOf("--only") + 1], "B");
assert.equal(run[run.indexOf("--limit") + 1], "2");
assert.equal(run[run.indexOf("--budget-cny") + 1], "5");

const bare = buildRunArgs({ remoteCid: "9", budgetFile: "l.json", budgetCny: 1 });
assert.ok(!bare.includes("--only"));
assert.ok(!bare.includes("--limit"));

console.log("args.test.mjs: all assertions passed");
