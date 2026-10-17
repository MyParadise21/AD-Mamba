"""Command-line audio tagging with the released AD-Mamba checkpoints."""
import argparse
import json
from pathlib import Path
import sys
from .predict import AudioClassifier


def main():
    parser = argparse.ArgumentParser(description="AD-Mamba AudioSet multi-label inference (first 1024 frames).")
    parser.add_argument("--audio", nargs="+", required=True, type=Path)
    parser.add_argument("--model", choices=("small", "medium"), default="small")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N; default: cpu")
    parser.add_argument("--scan-backend", choices=("torch", "oflex"), default="torch")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--all-scores", action="store_true", help="Output JSON including all 527 scores")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print JSON instead of a readable label table")
    parser.add_argument("--output", type=Path, help="Save results as UTF-8 JSON")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 527 or args.batch_size < 1 or args.threads < 1:
        parser.error("top-k must be 1..527; batch-size and threads must be positive.")
    try:
        classifier = AudioClassifier(args.model, args.checkpoint, args.device, args.scan_backend, args.threads)
        predictions = classifier.predict_many(args.audio, args.top_k, args.batch_size, args.all_scores)
        result = dict(model=args.model, checkpoint=str(classifier.checkpoint), device=str(classifier.device),
                      scan_backend=args.scan_backend, activation="sigmoid", predictions=predictions)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + chr(10), encoding="utf-8")
            print(f"Saved predictions to {args.output}", file=sys.stderr)
        elif args.json_output or args.all_scores:
            print(text)
        else:
            for prediction in predictions:
                print(f"\n{prediction['audio']}")
                print(f"{'Score':>8}  Label")
                for item in prediction["top_k"]:
                    print(f"{item['score']:8.4f}  {item['label']}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"AD-Mamba inference failed: {exc}" + chr(10))


if __name__ == "__main__":
    main()
