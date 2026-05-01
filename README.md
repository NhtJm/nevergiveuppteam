# Datathon 2026 Round 1 — Team Nevergiveup

This repository organizes the Round 1 submission into three competition parts plus the final report.

## Repository Structure

```text
.
├── part1/      # Multiple-choice answers and related notes
├── part2/      # Visualization and data analysis materials
├── part3/      # Sales forecasting pipeline, data, outputs, and submission files
└── report/     # NeurIPS-style report sections and report assets
```

## Part 1 — Multiple Choice

`part1/` is reserved for the answers to the 10 multiple-choice questions in the official submission form.

Current file:

- `part1/README.md`: overview of the MCQ notebook and answer mapping.
- `part1/part1.ipynb`: notebook used to compute/check Part 1 answers.

## Part 2 — Visualization and Data Analysis

`part2/` is reserved for EDA dashboards, figures, notebooks, and supporting files for the visualization and business analysis section.

Current file:

- `part2/README.md`: lightweight structure and working themes for Part 2.

The report should summarize Part 2 insights using the required structure:

- what each visualization shows,
- key findings supported by numbers,
- business implications or actionable recommendations.

## Part 3 — Sales Forecasting

`part3/` contains the complete forecasting pipeline for daily `Revenue` and `COGS`.

Key files:

- `part3/README.md`: detailed forecasting pipeline documentation.
- `part3/run.py`: CLI entry point for analysis, training, and explainability.
- `part3/src/`: source code for feature engineering, models, ensembling, tuning, and SHAP analysis.
- `part3/data/`: competition CSV files.
- `part3/outputs/submissions/submission_final.csv`: final Kaggle submission file.
- `part3/outputs/shap/`: SHAP plots and feature importance outputs.

To reproduce the forecasting output:

```bash
cd part3
pip install -r requirements.txt
python run.py train
python run.py explain
```

## Report

`report/` contains report writing material. The current report project is:

- `report/Styles/main.tex`: current NeurIPS-style report project.
- `report/report.zip`: original report archive.

The final report should follow the NeurIPS-style template required by the competition and stay within the 4-page limit excluding references and appendix.

## GitHub Repository

Public repository link:

<https://github.com/NhtJm/nevergiveuppteam>
