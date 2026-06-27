# AGENTS.md

Guide for AI agents working in this repository. Covers the non-obvious architecture, conventions, and gotchas that aren't apparent from reading a single file.

## Project overview

**learn-ai** ("图解 AI · 一行代码看懂一个概念") is a Chinese-language AI/ML tutorial published as a [VitePress](https://vitepress.dev) static site. 55 chapters cover ML foundations → classic ML → deep learning → CV → NLP → reinforcement learning → frontier topics (RAG/agents, multimodal, deployment, safety), plus a 16-section algorithms appendix. Each chapter pairs illustrated prose (`index.md`) with runnable Python code (`code/demo.py`) and a guided exercise (`code/exercise.py`).

- Site URL: deployed at `https://<user>.github.io/learn-ai/` (note the `/learn-ai/` base path).
- All content is written in **Chinese** (`lang: 'zh-CN'`); code comments are in Chinese. Keep new content in Chinese unless told otherwise.

## Essential commands

```bash
# Site (Node 20, see deploy.yml)
npm install
npm run dev        # local dev server
npm run build      # build to .vitepress/dist
npm run preview    # preview the built site

# Python code samples (CPU by default; GPU optional)
pip install -r requirements.txt
cd <chapter>/code && python demo.py        # e.g. s01_ai_overview/code
cd <chapter>/code && python exercise.py    # guided exercise with TODOs

# Special chapter with CLI flags
cd s16_attention_transformer/code && python nanogpt.py            # CPU ~15min
cd s16_attention_transformer/code && python nanogpt.py --gpu      # GPU
cd s16_attention_transformer/code && python nanogpt.py --generate # load & generate
```

There is **no test suite** and no linter configured. "Testing" a code change means running the relevant `demo.py` / `exercise.py` and confirming the output/figures look right. Verifying a docs change means `npm run build` (and ideally `npm run dev` to eyeball the page).

Deploy is automatic via `.github/workflows/deploy.yml` on push to `master`/`main` (GitHub Pages, `cancel-in-progress` concurrency).

## Critical architecture: code exists in THREE copies

This is the single most important gotcha. The same `demo.py` / `exercise.py` appears in three places that are **not auto-synced**:

| Location | Role | How it's used |
|----------|------|---------------|
| `<chapter>/code/*.py` | **Canonical source** — what users run locally | Referenced by README quick-start; run with `python demo.py` |
| `snippets/<chapter>/*.py` | **Embedded copy** for the rendered page | Pulled into `code-demo.md` / `code-exercise.md` via VitePress snippet include: `<<< @/snippets/<chapter>/demo.py` |
| `public/code/<chapter>/*.py` | **Static download copy** | Served at `/code/<chapter>/demo.py` and linked by the `<a href="../code/<chapter>/demo.py" download>` button at the top of each `code-demo.md` |

When you edit a chapter's code, update **all three** unless the task explicitly scopes to one. The download link in `code-demo.md` resolves as follows: the page lives at `/<chapter>/code-demo.html`, so `../code/<chapter>/demo.py` → `/code/<chapter>/demo.py`, which VitePress serves from `public/code/<chapter>/demo.py` (because `publicDir: 'public'`). The `@` alias in `<<< @/snippets/...` points to the project root (VitePress `srcDir`), so snippets resolve from `<root>/snippets/`.

The three copies have historically drifted (they are real files, not symlinks). Prefer editing `<chapter>/code/` first, then mirror to `snippets/` and `public/code/`. All `s*` chapters and all `ml*`/`algo*` chapters have `snippets/` entries; only the `s*` chapters currently have `public/code/` mirrors.

## VitePress configuration quirks (`.vitepress/config.mts`)

- **`base: '/learn-ai/'`** — every internal URL is prefixed with `/learn-ai/`. Keep this in mind when adding links; relative links in markdown are fine, but absolute paths must include the base.
- **`ignoreDeadLinks: true`** — broken links will **not** fail the build. This means `npm run build` succeeding does **not** prove links are valid. Check links manually when restructuring.
- **`srcExclude: ['README.md', '**/image_prompts.md', '**/CODE.md']`** — these files live in the repo but are **excluded from the built site**:
  - `README.md` is GitHub-only.
  - `image_prompts.md` files are internal image-generation notes (also gitignored via `**/image_prompts.md`).
  - **`CODE.md`** is a dev-only "code explanation + run report" working note that exists in every `s*` chapter (s01–s25) but is **never published** and is **not linked from `index.md`**. Treat it as a scratch/run-log, not user-facing content. `ml*` and `algo*` chapters do not have `CODE.md`.
- **Mermaid** is enabled via `vitepress-plugin-mermaid` — the config is wrapped in `withMermaid(defineConfig(...))`. The homepage `index.md` uses a `mermaid` fenced block for the learning roadmap. Use ```mermaid fences for diagrams.
- **Math** is on (`markdown.math: true`, backed by `markdown-it-mathjax3`). Use `$...$` for inline and `$$...$$` for display LaTeX. Every chapter's `index.md` relies heavily on math; don't strip it.
- **Code blocks show line numbers** (`markdown.lineNumbers: true`).
- Sidebar, nav, and outline are configured in `themeConfig`. The sidebar groups chapters into stages; the "番外" (extra) ML and the algorithms appendix groups are `collapsed: true` by default. When adding a chapter, register it in the sidebar here.

## Standard chapter structure

```
<chapter>/
├── index.md            # Main illustrated article (the core reading material)
├── code-demo.md        # Line-by-line walkthrough of demo.py; embeds snippet via <<<, has a Download button
├── code-exercise.md    # Exercise guide for exercise.py (TODOs to fill in)  [see exceptions below]
├── CODE.md             # Dev-only run report (s* chapters only; excluded from build)
├── code/
│   ├── demo.py         # Complete teaching implementation, Chinese comments
│   └── exercise.py     # Skeleton with TODO markers for the learner
└── images/             # Illustrations + generated output figures
```

**Exceptions to know about:**
- `s11b_vit/` (Vision Transformer) has **no `exercise.py` and no `code-exercise.md`** — only `vit_demo.py` and a `code-demo.md`. Don't assume exercise files exist.
- `s16_attention_transformer/` has an extra `nanogpt.py` (full GPT-2 from scratch) and a dedicated `nanogpt.md` walkthrough. The snippet/download mirrors for nanogpt live alongside demo/exercise.
- `s22_multimodal/images/samples/` ships real JPG samples (dog, cat, pizza, car) used by the CLIP demo.

Chapter ID prefixes encode the track: `sNN_` (main stages 1–7), `mlNN_` (classic + advanced ML), `algoNN_` (algorithms appendix). The sidebar order in `config.mts` is the source of truth for the intended reading order, not directory sorting.

## Code conventions

- Every script starts with `# -*- coding: utf-8 -*-` and a large triple-quoted docstring framed by `===` lines, summarising what the demo teaches and how to run it.
- **Figures are saved into the chapter's own `images/` directory using a script-relative path**, never a hardcoded absolute path:
  ```python
  _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  _IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
  os.makedirs(_IMAGES_DIR, exist_ok=True)
  ```
  Preserve this pattern when adding code that writes figures, so `python demo.py` works regardless of the caller's CWD.
- **Chinese font setup for matplotlib** is required or Chinese labels render as boxes. `exercise.py` files set:
  ```python
  matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```
  `demo.py` files typically set just `axes.unicode_minus = False`. Match the surrounding file's choice.
- **Fixed random seeds** (`np.random.seed(42)` / `torch.manual_seed`) are used throughout so output is reproducible — keep seeds stable when refactoring.
- Code is **CPU-first by default**; GPU is opt-in (see nanogpt `--gpu`). Demos must run on a consumer laptop.
- Comments explain *why* (intuition), not just *what*. The pedagogical style is "from scratch" — prefer NumPy-only implementations for foundational chapters (s01–s09), reach for PyTorch/transformers only where the chapter is about those frameworks.

## Image naming convention

Illustrations in `<chapter>/images/` follow `<prefix>-<NN>-<slug>.png`:
- `s*` chapters drop the `s`: `01-01-ai-ml-dl-relationship.png`, `17-01-bert-vs-gpt.png`.
- `ml*` and `algo*` keep the prefix: `ml01-01-knn-classification.png`, `algo01-01-big-o-omega-theta.png`.

Generated output figures (produced by `demo.py`) use plain slugs without the prefix: `perceptron_results.png`, `loss_curve.png`, `dqn_loss_curve.png`, etc. Keep these lowercase with underscores.

Alt text in `index.md` is descriptive Chinese (see the `![...](./images/...)` lines). Preserve descriptive alt text when adding images.

## Python environment

`requirements.txt` is a **single union list** covering all 25 chapters — install it once, not per chapter. Notable deps: `torch`/`torchvision`/`torchaudio`, `transformers`/`peft`/`trl`/`accelerate`/`bitsandbytes`, `vllm`, `faiss-cpu`/`chromadb`/`langchain`, `scikit-learn`, `opencv-python`, `librosa`, `openai`/`httpx`. Some chapters (s17, s18, s23) optionally call the OpenAI API or download HuggingFace models — see `.env.example` for `OPENAI_API_KEY` / `HF_TOKEN` / `LOCAL_MODEL_PATH` / `WANDB_API_KEY`. Copy to `.env` (gitignored) to use them; absence should degrade gracefully, not crash.

Downloaded models and training artifacts are gitignored: `**/models/`, `**/nanogpt_model.pt`, `**/input.txt`, `bert_sentiment_checkpoints/`, `data/`. Do not commit these.

## Git conventions

Recent commits use **Conventional Commits** in English: `docs:`, `chore:`, `feat:`, `fix:` (e.g. `docs: add 2 new learning paths`, `chore: add real images for ml11-ml14, replacing placeholders`). Some older commits use a `@ <type>:` Chinese form — don't reintroduce it. When committing, follow the English Conventional Commits style and keep the subject under 72 chars. Never commit `.claude/`, `.env`, build output, or downloaded models.

## Working effectively in this repo

- **Before editing a chapter's code**, check all three locations (`<chapter>/code/`, `snippets/<chapter>/`, `public/code/<chapter>/`) and decide whether the task needs all three updated.
- **Before editing `config.mts`**, remember `base`, `ignoreDeadLinks`, and `srcExclude` interact: a new file you add might be silently excluded or its dead links silently ignored.
- **`npm run build` passing ≠ correct.** Because `ignoreDeadLinks: true` and `srcExclude` hide files, always visually verify with `npm run dev` for content/link changes.
- **When adding a new chapter**, you must: create the directory + files above, add a `snippets/<chapter>/` copy, add a `public/code/<chapter>/` copy (if it's an `s*`-style chapter), and **register the chapter in the sidebar** in `.vitepress/config.mts`. Otherwise it won't appear in navigation.
- **Keep math and Mermaid intact** — many chapters' explanatory power depends on LaTeX derivations and flowcharts.
