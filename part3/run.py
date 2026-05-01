"""Single entry point — diagnostic + training + explainability."""
import sys
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

CMDS = {
    "analyze": "Diagnostic EDA on data",
    "tune":    "Optuna hyperparameter search (saves best_params.json)",
    "train":   "Train pipeline → submission_default.csv (FINAL)",
    "explain": "SHAP-based explainability (writes plots + CSVs)",
    "all":     "Run analyze → train → explain",
}


def usage():
    print("Usage: python run.py <command>")
    for k, v in CMDS.items():
        print(f"  {k:10s}  {v}")


def finalize_submission():
    """Copy submission_default.csv → submission_final.csv (the file to upload)."""
    src = Path("outputs/submissions/submission_default.csv")
    dst = Path("outputs/submissions/submission_final.csv")
    if src.exists():
        shutil.copy(src, dst)
        print(f"\n[FINAL] {dst} ready to upload to Kaggle.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        cmd = "all"
    else:
        cmd = sys.argv[1]

    if cmd == "analyze":
        import analyze; analyze.main()
    elif cmd == "tune":
        import tune
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        tune.main(n_trials=n)
    elif cmd == "train":
        import train; train.main()
        finalize_submission()
    elif cmd == "explain":
        import explain; explain.main()
    elif cmd == "all":
        print("=" * 70 + "\nSTEP 1: Diagnostic\n" + "=" * 70)
        import analyze; analyze.main()
        print("\n" + "=" * 70 + "\nSTEP 2: Training\n" + "=" * 70)
        import train; train.main()
        finalize_submission()
        print("\n" + "=" * 70 + "\nSTEP 3: Explainability\n" + "=" * 70)
        import explain; explain.main()
    else:
        usage()
        sys.exit(1)
