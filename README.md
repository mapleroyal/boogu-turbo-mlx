# boogu-turbo-mlx

`boogu-turbo-mlx` runs Boogu Image 0.1 Turbo text-to-image generation locally on
Apple Silicon with MLX. It downloads the official Hugging Face bf16 weights,
converts them into a local MLX artifact, and generates PNGs with embedded
metadata.

The default setup prepares a bf16 artifact. A q8 artifact is available as an
optional smaller runtime artifact after bf16 conversion.

## Requirements

- macOS on Apple Silicon.
- Python 3.10 or newer.
- Enough free disk space for model files:
  - Official source download: about 36 GB.
  - Default bf16 MLX artifact: about 33 GB.
  - Optional q8 artifact: about 21 GB.

Default bf16 setup needs roughly 70 GB if you keep the downloaded source, or
roughly 33 GB if you remove the source after artifact validation. Optional q8
setup can temporarily need close to 90 GB while it converts and quantizes. After
temporary files are cleaned, the final footprint depends on whether you keep the
downloaded source.

## Setup

Clone the repo, enter it, and run the shell script:

```bash
git clone https://github.com/mapleroyal/boogu-turbo-mlx.git
cd boogu-turbo-mlx
./setup.sh
```

The script creates `.venv`, installs the runtime and conversion tools, opens a
local setup page, and then continues in Terminal. The setup page lets you choose
the artifact type, source folder, artifact folders, memory mode, and cleanup
preferences.

If the configured source folder exists, setup reuses it. If it is missing and a
conversion needs source weights, setup downloads the source to that folder. To
reuse a source download from another clone or disk, set the source folder to
that existing `Boogu-Image-0.1-Turbo` path.

The setup page can also remove the project-local source folder after artifact
validation to save about 36 GB. If you remove it, preparing a new converted
artifact later, such as q8 after an initial bf16 setup, will require downloading
the source again or choosing an existing source folder.

If the browser does not open automatically:

```bash
./setup.sh --no-browser
```

Terminal will print a local URL you can open manually. To skip the setup page
and use the defaults:

```bash
./setup.sh --accept-defaults
```

The setup config is stored at `.boogu-turbo-mlx/config.json` and is ignored by
Git.

Setup also writes `Launch Boogu Turbo.command`, a local macOS launcher that
opens a Terminal log window and starts the browser GUI. The launcher is ignored
by Git because it is generated for your local checkout.

## Browser GUI

After setup, open `Launch Boogu Turbo.command` or run:

```bash
.venv/bin/boogu-turbo-mlx gui
```

The GUI reads `.boogu-turbo-mlx/config.json`, including the selected artifact,
output folder, and memory mode. In resident mode it preloads the model by
default; pass `--no-preload` if you want to open the GUI first and load the
model manually. In low-memory mode there is no resident preload; components are
loaded and released around generation.

The GUI binds to `127.0.0.1` on a random free port by default and opens a URL
with a per-session token. Binding to a non-loopback host requires
`--unsafe-host` and should only be used on a trusted network.

## Generate

After setup, generate an image with the default bf16 artifact:

```bash
.venv/bin/boogu-turbo-mlx generate \
  --model artifacts/boogu-mlx \
  --prompt "a glass library at sunrise" \
  --size 512x512 \
  --seed 42 \
  --output outputs/glass-library.png
```

For the optional q8 artifact:

```bash
.venv/bin/boogu-turbo-mlx generate \
  --model artifacts/boogu-mlx-q8 \
  --prompt "a glass library at sunrise" \
  --size 512x512 \
  --seed 42 \
  --output outputs/glass-library-q8.png
```

`generate` prints PNG paths on stdout. Progress and size notices are written to
stderr so scripts can safely read stdout.

All generation entry points require widths and heights to be positive multiples
of 16 up to 2048, and seeds must be integers from 0 to 4294967295.

## Batch Jobs

The browser GUI can run a declarative JSON queue file sequentially. Use a
top-level JSON array and save it as `.boogu-batch.json` or `.json`:

```json
[
  { "prompt": "a glass library at sunrise", "width": 768, "height": 768, "steps": 8, "seed": 1933305333 },
  { "prompt": "a cedar observatory at night", "width": 1024, "height": 1024, "steps": 6 }
]
```

Each job creates one image. `prompt`, `width`, `height`, and `steps` are
required. `seed` is optional; when omitted, the GUI assigns a random seed and
records that concrete value in the saved PNG metadata. Unknown keys are
rejected so typos fail early.

Widths and heights must follow the same GUI rules as single-image generation:
positive multiples of 16 up to 2048. Steps must be positive, seeds must be from
0 to 4294967295, and a queue can contain up to 100 jobs.

Jobs run in file order, one at a time. If a job fails, the batch stops there;
images from earlier successful jobs stay saved and visible in the gallery, and
later jobs are not attempted. The GUI provides a prompt to guide agents to create
a batch job file.

## Command Reference

Use `.venv/bin/boogu-turbo-mlx --version` to print the package version.
Every command also accepts `--help`.

`setup` configures, downloads, converts, validates, and optionally cleans up a
local artifact. `./setup.sh` creates the virtual environment, installs the
package, and forwards its flags to this command. The official source uses the
audited revision by default; pass `--source-revision` when using a different
remote source repo.

```bash
.venv/bin/boogu-turbo-mlx setup \
  [--config PATH] \
  [--source-repo REPO] [--source-revision REVISION] [--source-dir PATH] \
  [--artifact bf16|q8] \
  [--bf16-output PATH] [--q8-output PATH] [--q8-intermediate PATH] \
  [--output-dir PATH] \
  [--memory-mode resident|low-memory] \
  [--cleanup-intermediates | --keep-intermediates] \
  [--cleanup-source | --keep-source] \
  [--accept-defaults] [--no-browser] [--dry-run]
```

`gui` starts the local browser GUI:

```bash
.venv/bin/boogu-turbo-mlx gui \
  [--config PATH] [--host HOST] [--unsafe-host] [--port PORT] \
  [--no-browser] [--no-preload]
```

`manifest` inspects an official source folder or Hugging Face source without
loading large tensor shards:

```bash
.venv/bin/boogu-turbo-mlx manifest \
  --source PATH_OR_REPO [--revision REVISION] [--output PATH]
```

`download` fetches the official Hugging Face source, or another source when you
provide an explicit revision:

```bash
.venv/bin/boogu-turbo-mlx download \
  [--source REPO_OR_PATH] [--revision REVISION] [--dest PATH]
```

`convert` writes a bf16/fp artifact in the MLX layout:

```bash
.venv/bin/boogu-turbo-mlx convert \
  --source PATH_OR_REPO [--revision REVISION] \
  --output PATH \
  [--dtype auto|bfloat16|float16|float32]
```

`quantize` derives the persistent q8 artifact:

```bash
.venv/bin/boogu-turbo-mlx quantize \
  --source PATH --output PATH \
  [--bits 8] [--group-size 32] [--mode affine]
```

`generate` writes PNGs. Use `--output` for a single image and `--output-dir`
for repeated `--prompt` values or `--num-images-per-prompt` greater than 1.
`--output-template` applies only to batch output and can use
`{batch_index}`, `{prompt_index}`, `{image_index}`, and `{seed}`.

```bash
.venv/bin/boogu-turbo-mlx generate \
  --model PATH \
  --prompt TEXT [--prompt TEXT ...] \
  (--size WIDTHxHEIGHT | --height HEIGHT --width WIDTH) \
  (--output PATH | --output-dir PATH) \
  [--output-template TEMPLATE] \
  [--steps N] [--seed SEED | --seeds SEED,SEED,...] \
  [--timesteps T,T,...] \
  [--max-sequence-length N] [--truncate-instruction-sequence] \
  [--dmd-conditioning-sigma FLOAT] \
  [--num-images-per-prompt N] \
  [--progress auto|always|never] \
  [--memory-mode resident|low-memory] \
  [--denoise-batch-size N] [--decode-batch-size N]
```

`doctor` validates local environment, source, and artifact state. Use it before
debugging generation failures:

```bash
.venv/bin/boogu-turbo-mlx doctor \
  [--model PATH] [--source PATH] [--json]
```

`benchmark` measures generation latency and MLX memory:

```bash
.venv/bin/boogu-turbo-mlx benchmark \
  --model PATH \
  --prompt TEXT [--prompt TEXT ...] \
  --height HEIGHT --width WIDTH \
  [--steps N] [--seed SEED] [--timesteps T,T,...] \
  [--max-sequence-length N] [--truncate-instruction-sequence] \
  [--dmd-conditioning-sigma FLOAT] \
  [--num-images-per-prompt N] \
  [--runs N] [--warmup-runs N] [--json-output PATH] \
  [--memory-mode resident|low-memory] \
  [--denoise-batch-size N] [--decode-batch-size N]
```

## Runtime Controls

Generation supports:

```bash
--memory-mode resident|low-memory
--denoise-batch-size N
--decode-batch-size N
--progress auto|always|never
```

`resident` is the default fast path. `low-memory` loads major components
sequentially and can help on memory-constrained machines. Decode chunking can
reduce peak memory for batched generation:

```bash
.venv/bin/boogu-turbo-mlx generate \
  --model artifacts/boogu-mlx \
  --prompt "a glass library at sunrise" \
  --prompt "a cedar observatory under clear stars" \
  --num-images-per-prompt 2 \
  --size 512x512 \
  --seed 100 \
  --decode-batch-size 1 \
  --output-dir outputs/run1
```

Use `--seeds 10,11,12,13` for exact per-image seeds. Passing `--seed 100`
expands to `100, 101, ...` in prompt-major, image-minor order.

## Technical Notes

### Conversion

The downloaded Hugging Face source lives under `models/` unless you choose to
remove it after validation. Conversion creates a separate MLX artifact under
`artifacts/` and writes `artifact.json`, `manifest.json`, `weight_map.json`,
and `conversion_report.json` so the runtime can validate exactly what it is
loading.

The default conversion uses `--dtype auto`, which preserves the official source
dtypes. For the pinned Turbo source that means the main runtime path is bf16.
bf16 is the default because it is the fidelity baseline this project was built
around; q8 is an optional derived artifact, not the default experience.

Only tensors needed for text-to-image inference are copied:

- MLLM: text language-model weights are kept; vision-tower weights and `lm_head`
  are excluded.
- Transformer: text-to-image diffusion weights are kept; reference-image input
  weights are excluded.
- VAE: decoder weights are kept; encoder weights are excluded.

For the audited official source, that selects 398 MLLM tensors, 910 transformer
tensors, and 138 VAE tensors. The converter validates safetensors indexes and
headers before loading weights, preserves shard locality, and refuses to write
into a non-empty output folder unless it is already a valid artifact.

### q8 Quantization

`boogu-turbo-mlx quantize` creates a persistent MLX-native q8 artifact from the
bf16 artifact. The format is weight-only affine 8-bit quantization with group
size 32, implemented through MLX `nn.quantize`.

Quantization is deliberately narrow:

- Quantized: `nn.Linear` leaves in the MLLM text encoder and diffusion
  transformer.
- Not quantized: token embeddings, norms, non-linear parameters, scheduler,
  processor files, configs, and the VAE decoder.
- Rejected for this release: q4/q6, MXFP8 or direct FP8 import, activation
  quantization, VAE quantization, and runtime-only temporary quantization.

The q8 writer preflights every selected Linear input dimension for group-size
compatibility, stores quantization metadata in `artifact.json`, and writes a
`quantization_report.json`. Runtime loaders read that metadata, morph matching
Linear modules into MLX `QuantizedLinear`, then strict-load the indexed weights.

On the local validation artifact, q8 reduced the combined MLLM plus transformer
disk footprint to about 0.58x of bf16: MLLM 15.14 GB to 9.06 GB, transformer
19.87 GB to 11.18 GB. The q8 oracle compared identical prompt, seed, noise, and
timesteps against bf16 pre-VAE latents; full-pipeline drift measured cosine
0.999995, mean absolute error 0.003802, and max absolute error 0.025207.

### Runtime Decisions

`resident` is the default because it gives the fastest warm-path generation.
`low-memory` validates the artifact and then loads encoder, transformer, and VAE
sequentially. It is a memory-pressure option, not a speed option.

The bf16 benchmark matrix used 512x512, 4 Turbo steps, seed 0, one warmup run,
and three measured runs per case. On the validation machine:

- Resident single image: 2.115 seconds, 36.95 GiB peak.
- Low-memory single image: 3.216 seconds, 19.37 GiB peak.
- Resident batch of 2: 2.205 seconds per image, 40.00 GiB peak.
- Resident batch of 2 with `--decode-batch-size 1`: 2.044 seconds per image,
  36.95 GiB peak.

The runtime keeps the memory controls and chunking knobs because they changed
peak memory in useful ways. It also keeps always-on RoPE and attention-mask
hoisting because those are shape-invariant and preserve the same math.

Other speed paths were tested and removed rather than shipped as knobs:
manual fast-norm replacements, `mx.compile` over the denoise step, and manual
projection fusion. None cleared the speed gate for this 4-step Turbo workload;
the compile and normalization overhead did not amortize, and projection
microtiming was distorted by forced evaluation boundaries. Existing MLX fast
attention kernels are already used where applicable.

Advanced users can run the benchmark command directly:

```bash
.venv/bin/boogu-turbo-mlx benchmark \
  --model artifacts/boogu-mlx \
  --prompt "a glass library at sunrise" \
  --height 512 \
  --width 512 \
  --runs 3 \
  --warmup-runs 1
```

### Current Scope

This release targets local text-to-image inference for the official Boogu Image
0.1 Turbo source on Apple Silicon. CFG, reference-image inputs, direct FP8
artifact import, arbitrary community artifact compatibility, and non-Apple MLX
targets are outside the current runtime surface.

## Generated Folders

- `.venv/` holds the local Python environment.
- `.boogu-turbo-mlx/` holds local setup config and temporary setup files.
- `models/` holds explicit local Hugging Face downloads unless source cleanup
  removed them after validation.
- `artifacts/` holds converted or quantized MLX artifacts.
- `outputs/` holds generated PNGs.

These folders are ignored by Git.

## Python API

```python
from boogu_turbo_mlx import BooguTurboPipeline, save_generation_png

pipeline = BooguTurboPipeline.from_pretrained(
    "artifacts/boogu-mlx",
    memory_mode="resident",
)

batch = pipeline.generate_batch(
    ["a glass library at sunrise", "a cedar observatory under clear stars"],
    height=512,
    width=512,
    seed=100,
    num_images_per_prompt=2,
    decode_batch_size=1,
)

for item in batch:
    save_generation_png(
        item,
        f"outputs/image-{item.batch_index}.png",
        model="artifacts/boogu-mlx",
        memory_mode="resident",
        decode_batch_size=1,
    )
```

## Model And License Notes

This project is Apache-2.0 licensed. The setup command downloads
`Boogu/Boogu-Image-0.1-Turbo` from Hugging Face at the audited official
revision. The upstream model card also lists Apache-2.0; review the upstream
model card for current model-specific terms and usage notes.

Model downloads are explicit user actions. Importing the package, inspecting
metadata, and running the lightweight tests do not download or load large tensor
shards.

## Troubleshooting

- If setup cannot start, confirm you are on macOS with Apple Silicon and Python
  3.10 or newer.
- If `download` fails, confirm `huggingface-hub` is installed in `.venv` and the
  model is reachable.
- If conversion or quantization stops, check that the target artifact folder is
  empty or already contains a valid `artifact.json`.
- If generation fails, run `.venv/bin/boogu-turbo-mlx doctor --model artifacts/boogu-mlx`.
- If memory is tight, try `--memory-mode low-memory` or `--decode-batch-size 1`.
- Use `--progress always` for visible progress in logs and `--progress never`
  when quiet stderr matters.

## Manual Install

The setup script is the normal path. To install manually:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install ".[runtime,conversion]"
.venv/bin/boogu-turbo-mlx setup
```
