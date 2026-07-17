import { openSync } from "node:fs";
import { resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";

const workspace = resolve(".");
const powershell = "C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe";
const orchestrator = resolve(workspace, "analysis", "run_full_analysis.ps1");
const logDirectory = resolve(
  workspace,
  "results",
  "corrected_analysis_20260717",
  "orchestrator_logs",
);
const stdout = openSync(resolve(logDirectory, "launcher.stdout.log"), "a");
const stderr = openSync(resolve(logDirectory, "launcher.stderr.log"), "a");

const childArguments = [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    orchestrator,
    "-PythonExe",
    "tmp\\python311\\python.exe",
    "-DataPath",
    "Data_ML2.xlsx",
    "-ResultsDir",
    "results\\corrected_analysis_20260717",
    "-MaxParallel",
    "4",
    "-ModelThreads",
    "2",
  ];
const childOptions = {
    cwd: workspace,
    detached: true,
    windowsHide: true,
    stdio: ["ignore", stdout, stderr],
    env: {
      SystemRoot: "C:\\WINDOWS",
      WINDIR: "C:\\WINDOWS",
      COMSPEC: "C:\\WINDOWS\\System32\\cmd.exe",
      Path: [
        "C:\\WINDOWS\\System32",
        "C:\\WINDOWS",
        "C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0",
        "C:\\Program Files\\nodejs",
      ].join(";"),
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
      TEMP: resolve(workspace, "tmp"),
      TMP: resolve(workspace, "tmp"),
      USERPROFILE: "C:\\Users\\nhkwo",
      APPDATA: "C:\\Users\\nhkwo\\AppData\\Roaming",
      LOCALAPPDATA: "C:\\Users\\nhkwo\\AppData\\Local",
      TF_CPP_MIN_LOG_LEVEL: "2",
      MPLCONFIGDIR: resolve(workspace, "tmp", "mplconfig"),
      OMP_NUM_THREADS: "2",
    },
  };

if (process.argv.includes("--dry-run")) {
  const result = spawnSync(
    powershell,
    [...childArguments, "-DryRun"],
    { ...childOptions, detached: false, stdio: "inherit" },
  );
  process.exit(result.status ?? 1);
}

const child = spawn(powershell, childArguments, childOptions);

child.unref();
process.stdout.write(`${JSON.stringify({ pid: child.pid, orchestrator })}\n`);
