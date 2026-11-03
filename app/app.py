"""Two-tab Gradio demo for the Nepali NLP study."""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DISCLAIMER, EXAMPLES, FIGURE_PATH, FINAL_METRICS, GITHUB_URL, HELDOUT_TABLE,
    MATCHED_TABLE, REPORT_URL, XLMR_HELDOUT_STATUSES,
)
from model_loader import LoadedModel, load_all_models, predict

_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

try:
    import spaces
except ImportError:
    spaces = None


if spaces is not None:
    @spaces.GPU
    def _run_model(model: LoadedModel, text: str) -> dict[str, Any]:
        return predict(model, text)
else:
    def _run_model(model: LoadedModel, text: str) -> dict[str, Any]:
        return predict(model, text)


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _publisher_table(heldout: pd.DataFrame) -> str:
    xlmr_by_source = {}
    for path in XLMR_HELDOUT_STATUSES:
        status = json.loads(path.read_text(encoding="utf-8"))
        xlmr_by_source[status["held_out_source"]] = status["test_metrics"]["macro_f1"]
    lines = [
        "| Publisher | TF-IDF present-label macro-F1 | XLM-R matched-C macro-F1 |",
        "|---|---:|---:|",
    ]
    for row in heldout.itertuples(index=False):
        xlmr_value = xlmr_by_source.get(row.held_out_source)
        xlmr_text = _metric(xlmr_value) if xlmr_value is not None else "unavailable"
        lines.append(
            f"| {row.held_out_source} | {_metric(row.macro_f1_present_labels)} | {xlmr_text} |"
        )
    return "\n".join(lines)


def _research_markdown() -> str:
    final_data = json.loads(FINAL_METRICS.read_text(encoding="utf-8"))
    final = final_data[0] if isinstance(final_data, list) else final_data
    heldout = pd.read_csv(HELDOUT_TABLE)
    matched = pd.read_csv(MATCHED_TABLE)
    tfidf = heldout["macro_f1"].mean()
    tfidf_std = heldout["macro_f1"].std()
    xlmr = matched.loc[matched["model"].eq("XLM-RoBERTa-base"), "test_macro_f1"].iloc[0]
    return f"""## Research question

How well do Nepali news classifiers generalize when publisher/source changes,
and how do duplicate-aware evaluation, classical models, multilingual
embeddings, and a matched-budget transformer compare?

## Headline findings

- The frozen TF-IDF reference reached {_metric(final["accuracy"])} accuracy and
  {_metric(final["macro_f1"])} macro-F1 on the untouched test partition.
- The seven-publisher held-out TF-IDF diagnostic had mean present-label
  macro-F1 {_metric(tfidf)} (SD {_metric(tfidf_std)}).
- In the matched A comparison, XLM-R test macro-F1 was {_metric(xlmr)}.
- The LLM stage was a validation-only pilot and is not part of the main results.

## Held-out publisher comparison

The table below is loaded from saved TF-IDF and matched-C XLM-R status files.
The TF-IDF configuration is the bounded diagnostic; it is not the frozen
reference.

{_publisher_table(heldout)}

## Scope note

The XLM-R matched result and the held-out TF-IDF diagnostic use different
protocols and should not be read as a direct head-to-head score.

## LLM pilot (validation only, not part of the main results)

Two small 4-bit instruction models were tested on the shared validation
subset. The predeclared threshold was not met, so no test articles, few-shot
inference, or cross-publisher LLM evaluation was run.

## Limitations

The partitions are exact-duplicate-aware but not fully near-duplicate-safe;
labels are automatic and noisy; XLM-R uses one epoch and one seed; and the
cross-source analysis covers seven publishers only.
"""


def _model_card(model: LoadedModel, result: dict[str, Any] | None) -> str:
    if model.error:
        return f"### {model.display_name}\n\n**Unavailable:** `{html.escape(model.error)}`"
    if result is None:
        return f"### {model.display_name}\n\nNo prediction yet."
    top = "\n".join(f"- **{html.escape(label)}:** {score:.4f}" for label, score in result["top3"])
    uncertainty = " **UNCERTAIN**" if result["uncertain"] else ""
    return f"""### {model.display_name}

**Prediction:** {html.escape(result["label"])}{uncertainty}

**Score:** {result["confidence"]:.4f} ({html.escape(result["score_label"])})

**Top 3**
{top}

Latency: {result["latency"]:.3f}s
"""


def _optional_links() -> str:
    links = []
    if REPORT_URL and not REPORT_URL.startswith("<"):
        links.append(f"[Open the report]({REPORT_URL})")
    if GITHUB_URL and not GITHUB_URL.startswith("<"):
        links.append(f"[Open the GitHub repository]({GITHUB_URL})")
    return "  \n".join(links)


def _stats(text: str) -> str:
    length = len(text)
    devanagari = len(_DEVANAGARI.findall(text))
    warnings = []
    if length > 400:
        warnings.append("input exceeds the 400-token-style demo limit; neural models truncate it")
    if text and devanagari == 0:
        warnings.append("no Devanagari characters detected")
    warning_text = "; ".join(warnings) if warnings else "none"
    return f"Characters: {length}  \nDevanagari characters: {devanagari}  \nWarnings: {warning_text}"


def build_app(models: dict[str, LoadedModel] | None = None):
    import gradio as gr

    models = models or load_all_models()
    startup = "\n".join(
        f"- **{model.display_name}:** {'loaded' if not model.error else 'unavailable'} "
        f"({model.loaded_seconds:.2f}s)"
        for model in models.values()
    )
    state: dict[str, dict[str, Any] | None] = {key: None for key in models}

    def run_prediction(text: str):
        text = text or ""
        outputs = []
        results = []
        for key, model in models.items():
            if model.error:
                results.append(None)
                outputs.append(_model_card(model, None))
                continue
            try:
                result = _run_model(model, text)
            except Exception as exc:
                model.error = str(exc)
                result = None
            results.append(result)
            outputs.append(_model_card(model, result))
        available_labels = [r["label"] for r in results if r]
        agreement = (
            "Agreement: " + ", ".join(sorted(set(available_labels)))
            if available_labels else "Agreement unavailable."
        )
        return (*outputs, agreement, _stats(text))

    def clear():
        return "", *([""] * len(models)), "", ""

    with gr.Blocks(title="Nepali NLP Study Demo") as demo:
        gr.Markdown("# Nepali News Topic Classification")
        gr.Markdown(f"### Startup status\n{startup}")
        with gr.Tabs():
            with gr.Tab("Research"):
                gr.Markdown(_research_markdown())
                gr.Image(value=str(FIGURE_PATH), label="Held-out publisher macro-F1", type="filepath")
                links = _optional_links()
                if links:
                    gr.Markdown(links)
            with gr.Tab("Try it"):
                text = gr.Textbox(label="Nepali article text", lines=8, placeholder=EXAMPLES[0])
                with gr.Row():
                    predict_button = gr.Button("Predict", variant="primary")
                    clear_button = gr.Button("Clear")
                    example_button = gr.Button("Use example")
                stats = gr.Markdown()
                cards = [gr.Markdown(_model_card(model, None)) for model in models.values()]
                agreement = gr.Markdown()
                gr.Markdown(f"### Disclaimer\n{DISCLAIMER}")
                predict_button.click(run_prediction, inputs=text, outputs=[*cards, agreement, stats])
                clear_button.click(clear, outputs=[text, *cards, agreement, stats])
                example_button.click(lambda: EXAMPLES[0], outputs=text).then(
                    run_prediction, inputs=text, outputs=[*cards, agreement, stats]
                )
    return demo


if __name__ == "__main__":
    build_app().launch()
