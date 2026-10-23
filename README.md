# Offline Audio Privacy Scan

This project scans a local PCM WAV recording for short, broadband transient windows that may include keyboard sounds. It can write a JSON report and an optional audio copy with candidate windows silenced. It does not recover key identities or typed text, record from a microphone, generate masking noise, or send data to a service.

The detector uses short-time RMS level, high-band energy, and spectral flatness. You can adjust the minimum RMS threshold and per-file sensitivity; the high-band and flatness cutoffs are fixed heuristics. Those signals can occur in many ordinary sounds. Results are candidate intervals for review, not verified keystrokes, and missed events are possible.

## Requirements

- Python 3.10 or newer
- NumPy

Install the dependency:

```powershell
python -m pip install -r requirements.txt
```

## Scan a recording

```powershell
python -m src.audio_privacy "C:\Path\To\recording.wav"
```

Save a report and a redacted copy:

```powershell
python -m src.audio_privacy "C:\Path\To\recording.wav" `
  --report "reports\scan.json" `
  --redacted-output "reports\recording-redacted.wav"
```

Adjust the minimum frame level, adaptive sensitivity, or silence padding:

```powershell
python -m src.audio_privacy recording.wav --threshold-dbfs -42 --sensitivity-mad 5 --padding-ms 60
```

The CLI supports uncompressed integer PCM WAV at 8, 16, 24, or 32 bits per sample. Redacted output keeps the source sample format and silences each candidate interval plus the selected padding. Existing output paths are protected unless `--overwrite` is supplied. Run `python -m src.audio_privacy --help` for the options.

## Privacy and limits

- All processing is local and offline. No audio is recorded, uploaded, or included in the JSON report.
- The report contains source filename and SHA-256, audio metadata, interval timings, and detector thresholds; it does not contain audio samples or a transcript.
- A candidate window can be a key, speech, a click, or another transient. Listen and review before relying on the redacted copy.
- Silence from this heuristic does not establish that a recording contains no keyboard sounds or other sensitive audio.
