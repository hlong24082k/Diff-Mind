"""M3 Phase 1: merge a v1 LoRA adapter into the base model and export GGUFs (Kaggle, Unsloth).
Heavy imports (unsloth/torch/huggingface_hub) go inside main() so this module imports on the Mac venv for tests."""
import argparse
import json
import shutil
from pathlib import Path


def collect_ggufs(search_dirs: list[Path], name: str, out: Path, quants: list[str]) -> list[Path]:
    """Unsloth's GGUF filenames/output dir vary by version (`unsloth.Q4_K_M.gguf`, `<dir>_gguf/`...).
    Find the file for each requested quant and move it to `out/<name>-<quant>.gguf`."""
    found = [p for d in search_dirs if d.is_dir() for p in d.rglob("*.gguf")]
    moved = []
    for q in quants:
        hits = [p for p in found if q in p.name.lower()]
        assert len(hits) == 1, f"expected one {q} gguf, found {[str(p) for p in hits]} (all: {[str(p) for p in found]})"
        dest = out / f"{name}-{q}.gguf"
        shutil.move(hits[0], dest)
        moved.append(dest)
    return moved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="Hub repo id or local adapter dir")
    p.add_argument("--subfolder", default="", help="'' = Hub root (step 125); 'last-checkpoint' = step 201")
    p.add_argument("--name", required=True, help="output prefix, e.g. v1-s125")
    p.add_argument("--out", default="outputs/gguf")
    p.add_argument("--quant", nargs="+", default=["q4_k_m"], help="e.g. q4_k_m f16")
    p.add_argument("--expect-step", type=int, help="assert the checkpoint's trainer_state.json global_step")
    p.add_argument("--push-to", help="private Hub repo to upload the GGUFs to (HF_TOKEN from env)")
    args = p.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from unsloth import FastLanguageModel

    adapter = args.adapter
    if args.subfolder:  # Unsloth's from_pretrained has no `subfolder`, so fetch the folder and point at it
        adapter = Path(args.adapter) if Path(args.adapter).is_dir() else Path(
            snapshot_download(args.adapter, allow_patterns=f"{args.subfolder}/*"))
        adapter = str(adapter / args.subfolder)
    if args.expect_step is not None:
        step = json.loads(Path(adapter, "trainer_state.json").read_text())["global_step"]
        assert step == args.expect_step, f"{adapter} is step {step}, expected {args.expect_step}"
        print("checkpoint step:", step)

    # Same load as training: 4-bit base + adapter. Unsloth merges to 16-bit before converting.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter, max_seq_length=2048, dtype=torch.float16, load_in_4bit=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = out / f"{args.name}-merged"
    model.save_pretrained_gguf(str(merged), tok, quantization_method=args.quant)

    files = collect_ggufs([merged, Path(f"{merged}_gguf")], args.name, out, args.quant)
    for f in files:
        print(f"{f.stat().st_size / 2**30:6.2f} GiB  {f}")

    if args.push_to:
        api = HfApi()
        api.create_repo(args.push_to, private=True, exist_ok=True)
        for f in files:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=args.push_to)
        print("pushed", [f.name for f in files], "->", args.push_to)


if __name__ == "__main__":
    main()
