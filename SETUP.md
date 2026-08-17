# Setup and run — the single source of truth

Windows, PowerShell, from inside the `rag3gpp` folder. README.md and CLAUDE.md
both point here rather than repeating these commands, because three copies of
the same instructions is three things to keep in sync and they will drift.

## 1. Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **On the original dev machine the venv is one level UP**, at
> `RAG project\venv`, not inside `rag3gpp\`. That is a historical accident, and
> it is why `python -m src.ingest.download` fails with
> `No module named 'src'` if you run it from the parent folder: the venv is
> there but the package is here. Either activate `..\venv\Scripts\Activate.ps1`
> and `cd` into `rag3gpp`, or create a fresh venv here as above. Every command
> in this file assumes the working directory is `rag3gpp\`.

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

# Run order

Every module runs standalone with `python -m` and has a self-check or
`--sample` / `--explain` mode. Verify each stage before building the next on
it — a metadata bug found after 15 minutes of embedding costs 15 minutes.

## Offline: corpus → index

```powershell
python -m src.ingest.download --spec 38.331   # one spec first, proves the plumbing
python -m src.ingest.download                 # then the full Rel-18 corpus
python -m src.ingest.convert                  # normalise .doc -> .docx
python -m src.ingest.parse_docx               # clause-tree JSONL
python -m src.ingest.chunk                    # must report 0 orphaned conditions

python -m src.index.embedder --self-test      # proves GPU + fp16 + query prefix
python -m src.index.build --limit 200 --reset # smoke test
python -m src.index.build --reset             # full run, ~15 min on GPU
python -m src.index.bm25_store --build
```

### Two gates you must not skip

**`src.ingest.chunk` must report 0 orphaned conditions.** An orphan is a nested
`shall` that lost the `if` above it, which reads as an unconditional
requirement and is the most dangerous thing this corpus can produce. Do not
build an index on output that reports any.

**Every row of `data/raw/manifest.tsv` must say `Rel-18`.** A row that doesn't
means that spec has no Rel-18 version published and the downloader fell back to
the newest available, which silently breaks version pinning (Control #1).

## Online

```powershell
python -m src.retrieval.pipeline --query "When does the UE trigger T310?" --explain
python -m src.generation.answer --query "..."
python -m src.verification.groundedness --query "..."

python -m src.demo                            # all four controls, one per question

uvicorn src.api.main:app --port 8000
streamlit run src/ui/app.py
```

## Evaluation

```powershell
python -m eval.calibrate                      # fit the relevance-gate threshold
python -m eval.run_eval --retrieval           # ablation, GPU only, no API calls
python -m eval.run_eval --answers             # ablation, calls Gemini
python -m eval.run_eval --all
```

Re-run `eval.calibrate` after any change to the retriever, the reranker, or the
`*_FP16` flags — all of them move `RERANK_SCORE_THRESHOLD`.

## If something breaks

Paste me the full traceback. Common ones:

| Symptom | Likely cause |
|---|---|
| `403` or `Connection refused` from 3gpp.org | Corporate/college network blocking it — try mobile hotspot |
| `no versions found` | Spec number typo, or that spec moved series |
| `no .doc/.docx inside` | Zip contains only a PDF — we'll skip that spec |
| `LibreOffice not found` | Set `SOFFICE_PATH` in `.env` |
