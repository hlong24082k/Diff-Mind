"""M3 2.1: train/serve chat-template parity. Renders the GGUF's embedded chat template the way llama-cpp does at eval
time (Jinja2ChatFormatter, add_generation_prompt=True) and compares it byte-for-byte with the prompt part of the text
the model was trained on (outputs/v1/rendered_example.txt = train row 0, written by train_qlora --check).
  python -m src.evaluation.check_template --model ~/models/diff-mind/v1-s125-q4_k_m.gguf
Exit 0 = identical, 1 = mismatch (a unified diff is printed)."""
import argparse
import difflib
import sys

from src.data_pipeline.curate_dataset import read_jsonl
from src.data_pipeline.format_chat import build_messages

ASSISTANT = "<|im_start|>assistant\n"


def prompt_part(text: str) -> str:
    """Everything up to and including the generation prompt; drops the trained label + <|im_end|>."""
    return text[: text.index(ASSISTANT) + len(ASSISTANT)]


def render(template: str, messages: list[dict], eos: str = "", bos: str = "") -> str:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    return Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)(messages=messages).prompt


def compare(expected: str, got: str) -> str:
    """'' when identical, else a unified diff (repr per line so whitespace/newline differences are visible)."""
    if expected == got:
        return ""
    lines = lambda s: [repr(l) + "\n" for l in s.splitlines(keepends=True)]
    return "".join(difflib.unified_diff(lines(expected), lines(got), "trained (rendered_example.txt)", "gguf template"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the .gguf")
    ap.add_argument("--reference", default="outputs/v1/rendered_example.txt")
    ap.add_argument("--dataset", default="data/dataset", help="local dir; train row 0 is the rendered example")
    a = ap.parse_args(argv)

    from llama_cpp import Llama

    llm = Llama(model_path=a.model, vocab_only=True, verbose=False)  # tokenizer + metadata only, no weights
    tok = lambda i: llm.detokenize([i], special=True).decode() if i != -1 else ""
    eos, bos = tok(llm.token_eos()), tok(llm.token_bos())
    print(f"chat_format={llm.chat_format}  eos={eos!r}  bos={bos!r}")
    # Any other value means llama-cpp ignores the GGUF template (e.g. a built-in 'chatml' formatter) -> not what we check.
    assert llm.chat_format == "chat_template.default", f"llama-cpp uses {llm.chat_format!r}, not the GGUF template"

    row = read_jsonl(f"{a.dataset}/train.jsonl")[0]
    expected = prompt_part(open(a.reference).read())
    got = render(llm.metadata["tokenizer.chat_template"], build_messages(row["file_path"], row["diff"]), eos, bos)
    diff = compare(expected, got)
    print(diff or f"OK: prompts byte-identical ({len(got)} chars, row {row['id']})")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
