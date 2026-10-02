"""
Fine-tune TrOCR on MathWriting-2024 (run on a GPU, e.g. Google Colab).

  python finetune.py --train-limit 20000 --epochs 3 \
         --out models/finetuned

Training data comes ONLY from the official train folder and validation
data ONLY from the valid folder. The test folder is never touched here.
"""

import argparse
import glob
import inspect
import json
import os
import random

import numpy as np
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from common import parse_inkml, render_strokes, compute_metrics


class InkDataset(torch.utils.data.Dataset):
    def __init__(self, files, processor, pad_mode, max_len):
        self.files = files
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_mode = pad_mode
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        strokes, label = parse_inkml(self.files[i])
        img = render_strokes(strokes, pad_mode=self.pad_mode)

        pixel_values = self.processor(
            images=img, return_tensors="pt").pixel_values[0]

        ids = self.tok(
            label,
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        ).input_ids
        ids = [t if t != self.tok.pad_token_id else -100 for t in ids]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(ids),
        }


def pick(split_dir, limit, seed):
    files = sorted(glob.glob(os.path.join(split_dir, "*.inkml")))
    random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fhswf/TrOCR_Math_handwritten")
    ap.add_argument("--data-root",
                    default=os.path.join("data", "raw", "mathwriting-2024"))
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--pad", choices=["square", "stretch"], default="square")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join("models", "finetuned"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    tok = processor.tokenizer

    model.config.max_length = args.max_len
    model.generation_config.max_new_tokens = args.max_len
    model.generation_config.num_beams = 1

    train_files = pick(os.path.join(args.data_root, "train"),
                       args.train_limit, args.seed)
    val_files = pick(os.path.join(args.data_root, "valid"),
                     args.val_limit, args.seed)
    print("Train samples:", len(train_files), " Val samples:", len(val_files))

    train_ds = InkDataset(train_files, processor, args.pad, args.max_len)
    val_ds = InkDataset(val_files, processor, args.pad, args.max_len)

    def metrics_fn(pred):
        ids = pred.predictions
        if isinstance(ids, tuple):
            ids = ids[0]
        ids = np.where(ids < 0, tok.pad_token_id, ids)
        labels = np.where(pred.label_ids < 0, tok.pad_token_id, pred.label_ids)
        preds = tok.batch_decode(ids, skip_special_tokens=True)
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        m = compute_metrics(preds, refs)
        return {"cer": m["cer_corpus"], "exact_match": m["exact_match"]}

    # Argument name differs between transformers versions.
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strat = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        fp16=torch.cuda.is_available(),
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        seed=args.seed,
        **{strat: "steps"},
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=metrics_fn,
    )

    trainer.train()

    best_dir = os.path.join(args.out, "best")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)

    with open(os.path.join(args.out, "training_log.json"), "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    print("Best model saved to:", best_dir)


if __name__ == "__main__":
    main()