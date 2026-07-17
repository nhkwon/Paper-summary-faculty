param(
    [Parameter(Mandatory = $false)]
    [string]$WorkbookPath = "Data_ML2.xlsx"
)

$resolvedPath = (Resolve-Path -LiteralPath $WorkbookPath).Path
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-ZipEntryText {
    param(
        [System.IO.Compression.ZipArchive]$Archive,
        [string]$EntryName
    )

    $entry = $Archive.GetEntry($EntryName)
    if ($null -eq $entry) {
        return $null
    }

    $stream = $entry.Open()
    $reader = [System.IO.StreamReader]::new($stream)
    try {
        return $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Get-ColumnIndex {
    param([string]$CellReference)

    $letters = ($CellReference -replace '[^A-Z]', '')
    $index = 0
    foreach ($character in $letters.ToCharArray()) {
        $index = ($index * 26) + ([int][char]$character - [int][char]'A' + 1)
    }
    return $index - 1
}

function Get-Quantile {
    param(
        [double[]]$Values,
        [double]$Probability
    )

    if ($Values.Count -eq 0) {
        return [double]::NaN
    }

    $sorted = $Values | Sort-Object
    $position = ($sorted.Count - 1) * $Probability
    $lower = [math]::Floor($position)
    $upper = [math]::Ceiling($position)
    if ($lower -eq $upper) {
        return [double]$sorted[$lower]
    }

    $weight = $position - $lower
    return ([double]$sorted[$lower] * (1 - $weight)) + ([double]$sorted[$upper] * $weight)
}

$archive = [System.IO.Compression.ZipFile]::OpenRead($resolvedPath)
try {
    $sharedStrings = @()
    $sharedXmlText = Read-ZipEntryText -Archive $archive -EntryName 'xl/sharedStrings.xml'
    if ($sharedXmlText) {
        [xml]$sharedXml = $sharedXmlText
        foreach ($item in $sharedXml.SelectNodes("//*[local-name()='si']")) {
            $sharedStrings += $item.InnerText
        }
    }

    [xml]$workbookXml = Read-ZipEntryText -Archive $archive -EntryName 'xl/workbook.xml'
    $sheetNode = $workbookXml.SelectSingleNode("//*[local-name()='sheet'][1]")
    $sheetName = $sheetNode.GetAttribute('name')

    [xml]$sheetXml = Read-ZipEntryText -Archive $archive -EntryName 'xl/worksheets/sheet1.xml'
    $rows = @()
    foreach ($rowNode in $sheetXml.SelectNodes("//*[local-name()='sheetData']/*[local-name()='row']")) {
        $values = @{}
        foreach ($cellNode in $rowNode.SelectNodes("./*[local-name()='c']")) {
            $cellReference = $cellNode.GetAttribute('r')
            $columnIndex = Get-ColumnIndex -CellReference $cellReference
            $cellType = $cellNode.GetAttribute('t')
            $valueNode = $cellNode.SelectSingleNode("./*[local-name()='v']")
            $value = if ($null -eq $valueNode) { $null } else { $valueNode.InnerText }

            if ($cellType -eq 's' -and $null -ne $value) {
                $value = $sharedStrings[[int]$value]
            }
            elseif ($cellType -eq 'inlineStr') {
                $inlineNode = $cellNode.SelectSingleNode("./*[local-name()='is']")
                $value = if ($null -eq $inlineNode) { $null } else { $inlineNode.InnerText }
            }
            elseif ($null -ne $value -and $value -match '^-?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?$') {
                $value = [double]::Parse($value, [System.Globalization.CultureInfo]::InvariantCulture)
            }

            $values[$columnIndex] = $value
        }
        $rows += ,$values
    }

    $headers = @()
    $maxColumn = ($rows[0].Keys | Measure-Object -Maximum).Maximum
    for ($column = 0; $column -le $maxColumn; $column++) {
        $headers += [string]$rows[0][$column]
    }

    $dataRows = @(
        $rows | Select-Object -Skip 1 | Where-Object {
            @($_.Values | Where-Object { $null -ne $_ -and "$_" -ne '' }).Count -gt 0
        }
    )
    $summary = @()
    for ($column = 0; $column -le $maxColumn; $column++) {
        $columnValues = @($dataRows | ForEach-Object { $_[$column] })
        $nonMissing = @($columnValues | Where-Object { $null -ne $_ -and "$_" -ne '' })
        $numericValues = @($nonMissing | Where-Object { $_ -is [double] -or $_ -is [int] } | ForEach-Object { [double]$_ })
        $record = [ordered]@{
            column = $headers[$column]
            rows = $dataRows.Count
            missing = $dataRows.Count - $nonMissing.Count
            distinct = @($nonMissing | Sort-Object -Unique).Count
            data_type = if ($numericValues.Count -eq $nonMissing.Count) { 'numeric' } else { 'text' }
        }
        if ($numericValues.Count -eq $nonMissing.Count -and $numericValues.Count -gt 0) {
            $measure = $numericValues | Measure-Object -Minimum -Maximum -Average
            $record.min = [double]$measure.Minimum
            $record.q1 = Get-Quantile -Values $numericValues -Probability 0.25
            $record.mean = [double]$measure.Average
            $record.median = Get-Quantile -Values $numericValues -Probability 0.5
            $record.q3 = Get-Quantile -Values $numericValues -Probability 0.75
            $record.max = [double]$measure.Maximum
        }
        else {
            $record.levels = @($nonMissing | Group-Object | Sort-Object Count -Descending | ForEach-Object {
                [ordered]@{ value = $_.Name; count = $_.Count }
            })
        }
        $summary += [pscustomobject]$record
    }

    $targetColumn = [array]::IndexOf($headers, 'O')
    $targetValues = @($dataRows | ForEach-Object { $_[$targetColumn] } | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    $q1 = Get-Quantile -Values $targetValues -Probability 0.25
    $q3 = Get-Quantile -Values $targetValues -Probability 0.75
    $mean = ($targetValues | Measure-Object -Average).Average
    $thresholds = [ordered]@{
        Mean = [ordered]@{ value = $mean; low = @($targetValues | Where-Object { $_ -lt $mean }).Count; high = @($targetValues | Where-Object { $_ -ge $mean }).Count }
        Q3 = [ordered]@{ value = $q3; low = @($targetValues | Where-Object { $_ -lt $q3 }).Count; high = @($targetValues | Where-Object { $_ -ge $q3 }).Count }
        'Q3+IQR/2' = [ordered]@{ value = $q3 + (($q3 - $q1) / 2); low = @($targetValues | Where-Object { $_ -lt ($q3 + (($q3 - $q1) / 2)) }).Count; high = @($targetValues | Where-Object { $_ -ge ($q3 + (($q3 - $q1) / 2)) }).Count }
    }

    [ordered]@{
        workbook = $resolvedPath
        sheet = $sheetName
        row_count = $dataRows.Count
        column_count = $headers.Count
        columns = $headers
        column_summary = $summary
        full_dataset_thresholds = $thresholds
    } | ConvertTo-Json -Depth 8
}
finally {
    $archive.Dispose()
}
