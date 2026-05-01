# Part 1 — Multiple-Choice Questions

This folder contains the notebook used to compute the answers for the 10 multiple-choice questions in Datathon 2026 Round 1.

## Main File

- `part1.ipynb`: Colab notebook for loading the competition CSV files and computing each MCQ answer directly from the data.

## What the Notebook Does

The notebook follows a simple, auditable workflow:

1. Mount Google Drive in Colab.
2. Extract the official `datathon-2026-round-1.zip` archive.
3. Load the required CSV files with `pandas`.
4. Solve Q1-Q10 using direct joins, groupby operations, ratios, and summary statistics.

The notebook uses these core tables:

- `orders.csv`
- `order_items.csv`
- `products.csv`
- `customers.csv`
- `returns.csv`
- `web_traffic.csv`
- `payments.csv`
- `geography.csv`
- `sales.csv`

## Question Logic Summary

| Question | Computation in notebook | Result |
|---|---|---|
| Q1 | Median inter-order gap per customer from `orders.csv` | `144` days |
| Q2 | Average gross margin `(price - cogs) / price` by product segment | `Standard` |
| Q3 | Most common return reason for `Streetwear` products | `wrong_size` |
| Q4 | Lowest average bounce rate by traffic source | `email_campaign` |
| Q5 | Share of `order_items` rows with non-null `promo_id` | `38.66%` |
| Q6 | Average orders per customer by non-null age group | `55+` |
| Q7 | Revenue by region from delivered orders joined to geography | `East` |
| Q8 | Most common payment method among cancelled orders | `credit_card` |
| Q9 | Return-rate proxy by product size | `S` |
| Q10 | Highest average payment value by installment plan | `6` installments |

## Answer Mapping

Based on the official answer choices, the notebook outputs correspond to:

| Question | Answer |
|---|---|
| Q1 | C |
| Q2 | D |
| Q3 | B |
| Q4 | C |
| Q5 | C |
| Q6 | A |
| Q7 | C |
| Q8 | A |
| Q9 | A |
| Q10 | C |

## Reproducibility Note

The current notebook is written for Google Colab and uses paths under:

```text
/content/drive/MyDrive/Datathon/
```

If running locally inside this repository, update `base_path` to point to the CSV directory, for example:

```python
base_path = "../part3/data/"
```

