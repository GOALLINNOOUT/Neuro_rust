param(
    [string]$OutputDirectory = "data/squad"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $repoRoot $OutputDirectory
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$splits = @(
    @{ Name = 'train'; Url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' },
    @{ Name = 'validation'; Url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' }
)

foreach ($split in $splits) {
    $jsonPath = Join-Path $outputRoot "$($split.Name)-v1.1.json"
    $textPath = Join-Path $outputRoot "$($split.Name).txt"
    if (-not (Test-Path $jsonPath)) {
        Invoke-WebRequest -Uri $split.Url -OutFile $jsonPath
    }

    $dataset = Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $records = [System.Collections.Generic.List[string]]::new()
    foreach ($article in $dataset.data) {
        foreach ($paragraph in $article.paragraphs) {
            foreach ($qa in $paragraph.qas) {
                if ($qa.answers.Count -eq 0) { continue }
                $question = ([string]$qa.question) -replace '[\r\n]+', ' '
                $answerRecord = $qa.answers[0]
                $answer = ([string]$answerRecord.text) -replace '[\r\n]+', ' '
                if ([string]::IsNullOrWhiteSpace($question) -or [string]::IsNullOrWhiteSpace($answer)) { continue }
                $passage = [string]$paragraph.context
                $answerStart = [Math]::Max(0, [Math]::Min([int]$answerRecord.answer_start, $passage.Length))
                $answerEnd = [Math]::Min($passage.Length, $answerStart + $answer.Length)
                # Keep a compact evidence window around the annotated answer so
                # the example remains within the model's short context length.
                $windowStart = [Math]::Max(0, $answerStart - 80)
                while ($windowStart -gt 0 -and -not [char]::IsWhiteSpace($passage[$windowStart - 1])) { $windowStart-- }
                $windowEnd = [Math]::Min($passage.Length, $answerEnd + 80)
                while ($windowEnd -lt $passage.Length -and -not [char]::IsWhiteSpace($passage[$windowEnd])) { $windowEnd++ }
                $evidence = $passage.Substring($windowStart, $windowEnd - $windowStart) -replace '[\r\n]+', ' '
                $records.Add("Context: $evidence Question: $question Answer: $answer")
            }
        }
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($textPath, ($records -join "`n`n") + "`n", $encoding)
    Write-Output "$($split.Name): $($records.Count) question-answer examples -> $textPath"
}

$manifest = @"
Dataset: Stanford Question Answering Dataset (SQuAD) v1.1
Source: https://rajpurkar.github.io/SQuAD-explorer/
Dataset format: crowdworker questions paired with answer spans in Wikipedia passages.
License: CC BY-SA 4.0; attribution and share-alike terms apply to redistributed derivatives.
Preprocessing: each answerable question is written with a passage window of about 80 characters before and after the annotated answer: `Context: ... Question: ... Answer: ...`.
This teaches context-conditioned answer completion; it is not evidence retrieval or a guarantee of factual accuracy.
"@
[System.IO.File]::WriteAllText((Join-Path $outputRoot 'SOURCES.txt'), $manifest, (New-Object System.Text.UTF8Encoding($false)))
