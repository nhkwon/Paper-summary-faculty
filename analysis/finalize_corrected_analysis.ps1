param(
    [string]$PythonExe = "tmp\python311\python.exe",
    [string]$ResultsDir = "results\corrected_analysis_20260717"
)

$ErrorActionPreference = "Stop"
$pythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
$resultsPath = Join-Path (Resolve-Path -LiteralPath ".").Path $ResultsDir
$compiledPath = Join-Path $resultsPath "compiled"
$figuresPath = Join-Path $resultsPath "figures"

& $pythonPath "analysis\summarize_corrected_results.py" "--results-dir" $resultsPath "--bootstrap-resamples" "10000"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath "analysis\build_corrected_figures.py" "--compiled-dir" $compiledPath "--output-dir" $figuresPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath "analysis\build_manuscript_update.py" "--compiled-dir" $compiledPath "--output" "submission_notes\corrected_analysis_manuscript_updates.md"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath "analysis\build_corrected_notebook.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath "analysis\verify_pipeline.py" "--data" "Data_ML2.xlsx" "--output" (Join-Path $resultsPath "pipeline_unit_verification.json")
exit $LASTEXITCODE
