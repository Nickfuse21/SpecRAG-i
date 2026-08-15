# Setup — run these once

Windows, PowerShell, from inside the `rag3gpp` folder.

## 1. Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You should now see `(venv)` at the start of your prompt. **Everything below assumes it's active.**

## 2. PyTorch with CUDA — do this BEFORE requirements.txt

Order matters. If you install `sentence-transformers` first it pulls a CPU-only torch, and then your GPU sits idle while embedding takes 10× longer.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You must see `True` and your GPU name. If it says `False`, stop and tell me the output — don't continue, the whole pipeline assumes GPU.

> CUDA 12.1 wheels work on most current NVIDIA drivers. If it fails, run `nvidia-smi`, check the CUDA version in the top-right, and tell me what it says.

## 3. Everything else

```powershell
pip install -r requirements.txt
```

## 4. API key

```powershell
copy .env.example .env
```

Open `.env` and paste your Gemini key after `GEMINI_API_KEY=`. No quotes, no spaces.

## 5. LibreOffice

Needed only to convert older `.doc` specs to `.docx`. Install from
<https://www.libreoffice.org/download> — default options are fine.

If it doesn't end up on PATH, set the full path in `.env`:

```
SOFFICE_PATH=C:\Program Files\LibreOffice\program\soffice.exe
```

---

# Run the ingestion so far

```powershell
# one spec first, to prove the plumbing works end to end
python -m src.ingest.download --spec 38.331

# then the rest
python -m src.ingest.download

# normalise any .doc files to .docx
python -m src.ingest.convert
```

## What you should see

`data/raw/` fills with `.zip` files, `data/docx/` with `.docx`, and
`data/raw/manifest.tsv` lists exactly which version of each spec you pinned.

**Open that manifest and check it.** Every row should say `Rel-18`. If any row
shows a different release, that spec has no Rel-18 version published and the
downloader fell back to the newest available — which breaks version pinning
(Control #1). Tell me which spec and we'll decide whether to drop it.

## If something breaks

Paste me the full traceback. Common ones:

| Symptom | Likely cause |
|---|---|
| `403` or `Connection refused` from 3gpp.org | Corporate/college network blocking it — try mobile hotspot |
| `no versions found` | Spec number typo, or that spec moved series |
| `no .doc/.docx inside` | Zip contains only a PDF — we'll skip that spec |
| `LibreOffice not found` | Set `SOFFICE_PATH` in `.env` |
