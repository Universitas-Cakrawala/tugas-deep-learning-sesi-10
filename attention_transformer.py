"""Praktikum Sesi 10: attention dan transformer block.

Script ini hanya mencakup artefak untuk bagian Instruksi pada DL_Ass_10.pdf.
Jalankan dari root repository dengan:

    /usr/bin/python attention_transformer.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts"
SEED = 10
DTYPE = torch.float32


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _normalise_mask(mask: Tensor, scores: Tensor) -> Tensor:
    """Convert a boolean mask to the score shape and reject empty key rows."""
    if mask.dtype != torch.bool:
        raise TypeError("mask harus bertipe torch.bool")
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(0)
    try:
        expanded = torch.broadcast_to(mask, scores.shape)
    except RuntimeError as error:
        raise ValueError(f"shape mask {tuple(mask.shape)} tidak cocok dengan skor {tuple(scores.shape)}") from error
    if expanded.all(dim=-1).any():
        raise ValueError("setiap baris attention harus memiliki sedikitnya satu key yang valid")
    return expanded


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return output, scaled scores, and attention weights."""
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("dimensi query dan key harus sama")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("jumlah key dan value harus sama")
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(key.shape[-1])
    if mask is not None:
        expanded_mask = _normalise_mask(mask, scores)
        scores = scores.masked_fill(expanded_mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    output = weights @ value
    return output, scores, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model harus habis dibagi num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.last_shapes: dict[str, tuple[int, ...]] = {}
        self.last_attention_weights: Tensor | None = None

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, sequence_length, _ = tensor.shape
        return tensor.view(batch, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        key = query if key is None else key
        value = key if value is None else value
        q_projected = self.q_projection(query)
        k_projected = self.k_projection(key)
        v_projected = self.v_projection(value)
        q_heads = self._split_heads(q_projected)
        k_heads = self._split_heads(k_projected)
        v_heads = self._split_heads(v_projected)
        output_heads, _, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        concatenated = output_heads.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.d_model
        )
        output = self.output_projection(concatenated)
        self.last_attention_weights = weights.detach()
        self.last_shapes = {
            "q_before_split": tuple(q_projected.shape),
            "k_before_split": tuple(k_projected.shape),
            "v_before_split": tuple(v_projected.shape),
            "q_after_split": tuple(q_heads.shape),
            "k_after_split": tuple(k_heads.shape),
            "v_after_split": tuple(v_heads.shape),
            "scores_per_head": tuple(weights.shape),
            "weights_per_head": tuple(weights.shape),
            "heads_before_concat": tuple(output_heads.shape),
            "after_concat": tuple(concatenated.shape),
            "mha_output": tuple(output.shape),
        }
        return output


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 128) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=DTYPE).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2, dtype=DTYPE) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=DTYPE)
        encoding[:, 0::2] = torch.sin(positions * denominator)
        encoding[:, 1::2] = torch.cos(positions * denominator)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("input positional encoding harus berbentuk (B, T, d_model)")
        sequence_length = inputs.shape[1]
        if sequence_length > self.encoding.shape[1]:
            raise ValueError("panjang sekuens melebihi max_length positional encoding")
        return inputs + self.encoding[:, :sequence_length, :].to(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.last_shapes: dict[str, tuple[int, ...]] = {}

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        attention_output = self.attention(inputs, mask=mask)
        attention_residual = self.attention_norm(inputs + attention_output)
        feed_forward_output = self.feed_forward(attention_residual)
        output = self.feed_forward_norm(attention_residual + feed_forward_output)
        self.last_shapes = {
            "input": tuple(inputs.shape),
            **self.attention.last_shapes,
            "attention_residual_layer_norm": tuple(attention_residual.shape),
            "ffn_output": tuple(feed_forward_output.shape),
            "block_output": tuple(output.shape),
        }
        return output


class SequenceClassifier(nn.Module):
    def __init__(self, d_model: int = 8, num_heads: int = 2, d_ff: int = 32) -> None:
        super().__init__()
        self.position = SinusoidalPositionalEncoding(d_model)
        self.block = TransformerBlock(d_model, num_heads, d_ff)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.position(inputs)
        transformed = self.block(encoded)
        return self.classifier(transformed.mean(dim=1))


def manual_attention_verification() -> dict[str, object]:
    query = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=DTYPE)
    keys = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=DTYPE)
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    expected_weights = torch.softmax(torch.tensor([0.5, 0.0], dtype=DTYPE), dim=-1)
    expected_output = expected_weights @ values
    output, scores, weights = scaled_dot_product_attention(query, keys, values)
    max_error = (output - expected_output).abs().max().item()
    assert torch.allclose(output, expected_output, atol=1e-6)
    assert torch.allclose(weights.squeeze(0).sum(), torch.tensor(1.0), atol=1e-6)
    return {
        "raw_scores": [1.0, 0.0],
        "scaled_scores": scores.squeeze(0).tolist(),
        "manual_weights": expected_weights.tolist(),
        "program_weights": weights.squeeze(0).tolist(),
        "manual_output": expected_output.tolist(),
        "program_output": output.squeeze(0).tolist(),
        "max_abs_error": max_error,
        "passed": True,
    }


def masking_verification(mha: MultiHeadAttention, inputs: Tensor) -> dict[str, object]:
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    mha(inputs, mask=padding_mask[:, None, None, :])
    assert mha.last_attention_weights is not None
    padding_weights = mha.last_attention_weights
    padding_max = padding_weights.masked_select(padding_mask[:, None, None, :]).abs().max().item()
    valid_row_sums = padding_weights.sum(dim=-1)
    assert padding_max <= 1e-6
    assert torch.allclose(valid_row_sums, torch.ones_like(valid_row_sums), atol=1e-6)

    causal_mask = torch.triu(torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool), diagonal=1)
    mha(inputs, mask=causal_mask)
    assert mha.last_attention_weights is not None
    causal_weights = mha.last_attention_weights
    causal_max = causal_weights.masked_select(causal_mask).abs().max().item()
    assert causal_max <= 1e-6
    assert torch.allclose(causal_weights.sum(dim=-1), torch.ones_like(causal_weights.sum(dim=-1)), atol=1e-6)
    rejected_empty_row = False
    try:
        scaled_dot_product_attention(inputs[:, :1, :4], inputs[:, :1, :4], inputs[:, :1, :4],
                                     mask=torch.ones(1, 1, dtype=torch.bool))
    except ValueError:
        rejected_empty_row = True
    assert rejected_empty_row
    return {
        "padding_mask": padding_mask.tolist(),
        "padding_forbidden_max_weight": padding_max,
        "causal_mask": causal_mask.tolist(),
        "causal_forbidden_max_weight": causal_max,
        "padding_row_sums": valid_row_sums.tolist(),
        "causal_row_sums": causal_weights.sum(dim=-1).tolist(),
        "empty_key_row_rejected": rejected_empty_row,
        "passed": True,
    }


def shape_and_parameter_verification(inputs: Tensor) -> dict[str, object]:
    d_model, num_heads, d_ff = 8, 2, 32
    mha = MultiHeadAttention(d_model, num_heads)
    block = TransformerBlock(d_model, num_heads, d_ff)
    mha_output = mha(inputs)
    block_output = block(inputs)
    expected_mha_parameters = 288
    expected_block_parameters = 872
    assert count_parameters(mha) == expected_mha_parameters
    assert count_parameters(block) == expected_block_parameters
    expected_shapes = {
        "input": (2, 4, 8),
        "q_before_split": (2, 4, 8),
        "k_before_split": (2, 4, 8),
        "v_before_split": (2, 4, 8),
        "q_after_split": (2, 2, 4, 4),
        "k_after_split": (2, 2, 4, 4),
        "v_after_split": (2, 2, 4, 4),
        "scores_per_head": (2, 2, 4, 4),
        "weights_per_head": (2, 2, 4, 4),
        "heads_before_concat": (2, 2, 4, 4),
        "after_concat": (2, 4, 8),
        "mha_output": (2, 4, 8),
        "attention_residual_layer_norm": (2, 4, 8),
        "ffn_output": (2, 4, 8),
        "block_output": (2, 4, 8),
    }
    actual_shapes = {"input": tuple(inputs.shape), **block.last_shapes}
    assert actual_shapes == expected_shapes
    assert tuple(mha_output.shape) == (2, 4, 8)
    assert tuple(block_output.shape) == (2, 4, 8)
    return {
        "shapes": {name: list(shape) for name, shape in actual_shapes.items()},
        "mha_parameters_manual": expected_mha_parameters,
        "mha_parameters_program": count_parameters(mha),
        "block_parameters_manual": expected_block_parameters,
        "block_parameters_program": count_parameters(block),
        "passed": True,
    }


def positional_encoding_verification(inputs: Tensor) -> dict[str, object]:
    positional_encoding = SinusoidalPositionalEncoding(d_model=8, max_length=4)
    output = positional_encoding(inputs)
    positions = positional_encoding.encoding.squeeze(0)
    assert tuple(positions.shape) == (4, 8)
    assert tuple(output.shape) == tuple(inputs.shape)
    assert not torch.allclose(positions[0], positions[1])
    return {
        "encoding_shape": list(positions.shape),
        "broadcast_output_shape": list(output.shape),
        "first_two_positions": positions[:2].tolist(),
        "passed": True,
    }


def make_synthetic_data(num_samples: int, sequence_length: int = 4) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(SEED + num_samples + sequence_length)
    inputs = torch.randn(num_samples, sequence_length, 8, generator=generator)
    labels = (inputs[:, :, 0].mean(dim=1) > 0).long()
    return inputs, labels


def train_functional_model() -> dict[str, object]:
    train_inputs, train_labels = make_synthetic_data(32)
    test_inputs, test_labels = make_synthetic_data(64)
    model = SequenceClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = loss_function(logits, train_labels)
        loss.backward()
        optimizer.step()
        accuracy = (logits.argmax(dim=-1) == train_labels).float().mean().item()
        history.append({"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})
    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs)
        test_accuracy = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()
    final = history[-1]
    assert final["accuracy"] >= 0.95
    assert final["loss"] < history[0]["loss"]
    return {
        "epochs": len(history),
        "initial_loss": history[0]["loss"],
        "final_loss": final["loss"],
        "final_train_accuracy": final["accuracy"],
        "test_accuracy": test_accuracy,
        "overfit_one_batch_passed": True,
        "history": history,
    }


def write_outputs(results: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    serialisable = json.loads(json.dumps(results))
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )
    training_history = results["functional_test"]["history"]
    with (OUTPUT_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "accuracy"])
        writer.writeheader()
        writer.writerows(training_history)

    summary = [
        "# Verification Report",
        "",
        "Artefak ini hanya mencakup bagian **Instruksi**, bukan Pertanyaan Analisis.",
        "",
        f"- Manual attention: {'PASS' if results['manual_attention']['passed'] else 'FAIL'}",
        f"- Shape dan parameter: {'PASS' if results['shape_and_parameters']['passed'] else 'FAIL'}",
        f"- Masking: {'PASS' if results['masking']['passed'] else 'FAIL'}",
        f"- Positional encoding: {'PASS' if results['positional_encoding']['passed'] else 'FAIL'}",
        f"- Overfit satu batch: {'PASS' if results['functional_test']['overfit_one_batch_passed'] else 'FAIL'}",
        "",
        "## Parameter",
        "",
        "| Komponen | Manual | Program |",
        "|---|---:|---:|",
        f"| Multi-head attention | {results['shape_and_parameters']['mha_parameters_manual']} | {results['shape_and_parameters']['mha_parameters_program']} |",
        f"| Transformer block | {results['shape_and_parameters']['block_parameters_manual']} | {results['shape_and_parameters']['block_parameters_program']} |",
        "",
        "## Functional test",
        "",
        f"- Initial loss: `{results['functional_test']['initial_loss']:.6f}`",
        f"- Final loss: `{results['functional_test']['final_loss']:.6f}`",
        f"- Final train accuracy: `{results['functional_test']['final_train_accuracy']:.4f}`",
        f"- Test accuracy: `{results['functional_test']['test_accuracy']:.4f}`",
        "",
        "Detail numerik tersedia di `verification_report.json` dan `training_metrics.csv`.",
    ]
    (OUTPUT_DIR / "verification_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    set_seed()
    inputs = torch.randn(2, 4, 8, dtype=DTYPE)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    results: dict[str, object] = {
        "manual_attention": manual_attention_verification(),
        "shape_and_parameters": shape_and_parameter_verification(inputs),
        "masking": masking_verification(mha, inputs),
        "positional_encoding": positional_encoding_verification(inputs),
        "functional_test": train_functional_model(),
    }
    write_outputs(results)
    print(json.dumps({key: value for key, value in results.items() if key != "functional_test"}, indent=2))
    print(
        "functional_test:",
        json.dumps({key: value for key, value in results["functional_test"].items() if key != "history"}, indent=2),
    )
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
