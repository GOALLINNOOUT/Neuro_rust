param(
    [int]$TrainRows = 2000,
    [int]$ValidationRows = 200,
    [int]$BlimpPairsPerCategory = 10
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $repoRoot 'data'
$tinyRoot = Join-Path $dataRoot 'tinystories'
$blimpRoot = Join-Path $dataRoot 'blimp'
New-Item -ItemType Directory -Force -Path $tinyRoot, $blimpRoot | Out-Null

function Get-TinyStoriesSample {
    param([string]$Split, [int]$Count, [int]$TotalRows, [string]$OutputPath)
    if (Test-Path $OutputPath) {
        return ([System.IO.File]::ReadAllText($OutputPath) -split "`n`n").Where({ -not [string]::IsNullOrWhiteSpace($_) }).Count
    }
    $pageSize = 100
    $pages = [Math]::Ceiling($Count / $pageSize)
    $storyList = [System.Collections.Generic.List[string]]::new()
    for ($page = 0; $page -lt $pages; $page++) {
        $offset = if ($pages -le 1) { 0 } else { [int][Math]::Floor($page * ($TotalRows - $pageSize) / ($pages - 1)) }
        $length = [Math]::Min($pageSize, $Count - $storyList.Count)
        $query = "dataset=roneneldan%2FTinyStories&config=default&split=$Split&offset=$offset&length=$length"
        $response = $null
        for ($attempt = 1; $attempt -le 5 -and $null -eq $response; $attempt++) {
            try {
                $response = Invoke-RestMethod -Uri "https://datasets-server.huggingface.co/rows?$query" -TimeoutSec 90
            } catch {
                if ($attempt -eq 5) { throw }
                Start-Sleep -Seconds ([Math]::Min(2 * $attempt, 10))
            }
        }
        foreach ($entry in $response.rows) {
            $story = [string]$entry.row.text
            $story = ($story -replace "`r?`n", ' ').Trim()
            if ($story.Length -gt 0) { $storyList.Add($story) }
        }
        Write-Progress -Activity "Downloading TinyStories $Split sample" -Status "$($storyList.Count) / $Count stories" -PercentComplete (100 * $storyList.Count / $Count)
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($OutputPath, ($storyList -join "`n`n") + "`n", $encoding)
    Write-Progress -Activity "Downloading TinyStories $Split sample" -Completed
    return $storyList.Count
}

$trainCount = Get-TinyStoriesSample -Split 'train' -Count $TrainRows -TotalRows 2119719 -OutputPath (Join-Path $tinyRoot 'train.txt')
$validCount = Get-TinyStoriesSample -Split 'validation' -Count $ValidationRows -TotalRows 21990 -OutputPath (Join-Path $tinyRoot 'validation.txt')

$zipPath = Join-Path $blimpRoot 'BLiMP.zip'
$extractRoot = Join-Path $blimpRoot 'source'
$tsvPath = Join-Path $blimpRoot 'blimp-sample.tsv'
if (-not (Test-Path $zipPath)) {
    Invoke-WebRequest -Uri 'https://github.com/alexwarstadt/blimp/raw/refs/heads/master/BLiMP.zip' -OutFile $zipPath
}
if (-not (Test-Path $extractRoot)) {
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force
}

$records = [System.Collections.Generic.List[string]]::new()
Get-ChildItem -LiteralPath $extractRoot -Filter '*.jsonl' -File -Recurse | Where-Object { $_.Name -notlike '._*' } | Sort-Object FullName | ForEach-Object {
    $category = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $taken = 0
    foreach ($line in [System.IO.File]::ReadLines($_.FullName)) {
        if ($taken -ge $BlimpPairsPerCategory) { break }
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $pair = $line | ConvertFrom-Json
        $good = ([string]$pair.sentence_good) -replace "`t|`r|`n", ' '
        $bad = ([string]$pair.sentence_bad) -replace "`t|`r|`n", ' '
        $records.Add("$category`t$good`t$bad")
        $taken++
    }
}
$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($tsvPath, $records, $encoding)

$manifest = @"
TinyStories source: https://huggingface.co/datasets/roneneldan/TinyStories
TinyStories license: CDLA-Sharing-1.0; see the dataset card before redistributing.
TinyStories split rows downloaded: train=$trainCount, validation=$validCount
Sampling: evenly spaced row windows from the official train and validation splits.

BLiMP source: https://github.com/alexwarstadt/blimp
BLiMP license: CC-BY; cite Warstadt et al. (2020).
BLiMP pairs included: $($records.Count) (up to $BlimpPairsPerCategory from each paradigm).
"@
[System.IO.File]::WriteAllText((Join-Path $dataRoot 'SOURCES.txt'), $manifest, $encoding)
Write-Output "TinyStories rows: train=$trainCount validation=$validCount"
Write-Output "BLiMP paradigms sampled: $($records.Count) pairs"
