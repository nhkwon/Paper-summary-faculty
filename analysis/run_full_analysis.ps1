param(
    [string]$PythonExe = "tmp\python311\python.exe",
    [string]$DataPath = "Data_ML2.xlsx",
    [string]$ResultsDir = "results\corrected_analysis_20260717",
    [int]$MaxParallel = 4,
    [int]$ModelThreads = 2,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path -LiteralPath ".").Path
$pythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
$dataFile = (Resolve-Path -LiteralPath $DataPath).Path
$pipelinePath = Join-Path $workspace "analysis\corrected_cost_pipeline.py"
$resultsPath = Join-Path $workspace $ResultsDir
$logsPath = Join-Path $resultsPath "orchestrator_logs"
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$transcriptPath = Join-Path $logsPath "orchestrator.transcript.log"
Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null

$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:MPLCONFIGDIR = Join-Path $workspace "tmp\mplconfig"
$env:OMP_NUM_THREADS = "$ModelThreads"

$featureSets = @("A_all_21", "B_no_component_costs", "C_physical_quantity")
$seeds = 42..51
$queue = [System.Collections.Generic.Queue[object]]::new()

foreach ($featureSet in $featureSets) {
    foreach ($seed in $seeds) {
        $taskName = "${featureSet}_seed${seed}"
        $completionPath = Join-Path $resultsPath "tasks\$taskName\complete.json"
        if ((Test-Path -LiteralPath $completionPath) -and -not $Force) {
            Write-Host "SKIP $taskName (already complete)"
            continue
        }
        $queue.Enqueue([pscustomobject]@{
            Name = $taskName
            FeatureSet = $featureSet
            Seed = $seed
        })
    }
}

if ($DryRun) {
    Write-Host "Dry run: $($queue.Count) task(s) queued."
    Stop-Transcript | Out-Null
    exit 0
}

$running = @()
$finished = @()
$failures = @()

function Start-AnalysisTask {
    param([pscustomobject]$Task)

    $arguments = @(
        $pipelinePath,
        "--data", $dataFile,
        "--output-dir", $resultsPath,
        "--seeds", "$($Task.Seed)",
        "--feature-sets", $Task.FeatureSet,
        "--grid-jobs", "1",
        "--model-threads", "$ModelThreads"
    )
    if ($Force) {
        $arguments += "--force"
    }
    $quotedArguments = $arguments | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processStartInfo.FileName = $pythonPath
    $processStartInfo.Arguments = $quotedArguments -join " "
    $processStartInfo.WorkingDirectory = $workspace
    $processStartInfo.UseShellExecute = $false
    $processStartInfo.CreateNoWindow = $true
    $processStartInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processStartInfo
    if (-not $process.Start()) {
        throw "Failed to start $($Task.Name)"
    }
    Write-Host "START $($Task.Name) pid=$($process.Id)"
    return [pscustomobject]@{
        Task = $Task
        Process = $process
        StartedAt = Get-Date
        Stdout = $transcriptPath
        Stderr = $transcriptPath
    }
}

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $running += Start-AnalysisTask -Task $queue.Dequeue()
    }

    Start-Sleep -Seconds 2
    $stillRunning = @()
    foreach ($entry in $running) {
        $entry.Process.Refresh()
        if ($entry.Process.HasExited) {
            $entry.Process.WaitForExit()
            $elapsed = (Get-Date) - $entry.StartedAt
            $record = [pscustomobject]@{
                Task = $entry.Task.Name
                ExitCode = $entry.Process.ExitCode
                ElapsedMinutes = [math]::Round($elapsed.TotalMinutes, 2)
                Stdout = $entry.Stdout
                Stderr = $entry.Stderr
            }
            $finished += $record
            if ($entry.Process.ExitCode -eq 0) {
                Write-Host "DONE  $($entry.Task.Name) in $($record.ElapsedMinutes) min"
            }
            else {
                Write-Host "FAIL  $($entry.Task.Name) exit=$($entry.Process.ExitCode)"
                $failures += $record
            }
        }
        else {
            $stillRunning += $entry
        }
    }
    $running = $stillRunning
}

$manifestPath = Join-Path $logsPath "orchestrator_manifest.csv"
$finished | Sort-Object Task | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
Write-Host "Manifest: $manifestPath"

if ($failures.Count -gt 0) {
    Write-Host "$($failures.Count) task(s) failed."
    Stop-Transcript | Out-Null
    exit 1
}

Write-Host "All queued tasks completed successfully."
Stop-Transcript | Out-Null
exit 0
