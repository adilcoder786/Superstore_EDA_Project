# Superstore Retail Sales: Exploratory Data Analysis

Internship assignment (Datheon). Complete EDA on the Superstore sales dataset: cleaning, feature engineering, customer / category / region / discount / seasonality analysis, 12 charts, business insights and an executive summary.

## Headline results (all calculated from the data)
- Sales **\$2.30M**, profit **\$286.4K** (12.5% margin), 5,009 orders, 793 customers, 2015-2018.
- Lines discounted **above 20% lost \$135.4K** in total; every discount band above 20% is loss-making.
- **Furniture** = 32% of sales but only 6% of profit (Tables and Bookcases lose money).
- **Sep-Dec = 51.6% of sales**, and the same three peak months (Sep, Nov, Dec) repeat in all 4 years.
- **Central region** has the lowest margin (7.9% vs 14.9% in West) and the heaviest discounting.

## Data quality finding
The uploaded CSV contained 10,800 rows, but only **9,994 are real sales**. 806 rows were pieces of two other sheets (a *People* table and a *Returns* table) appended to the file. They were separated (not blindly deleted): the returned Order IDs were reused for a returns analysis. 1 duplicate sales line was removed, and 11 missing Postal Codes were filled (Burlington, VT = 05401). Outliers were investigated and kept (flagged), because they are genuine.

## Folder contents
| File | What it is |
|---|---|
| `Superstore_EDA.ipynb` | Full notebook, 17 required sections + export appendix, already executed with outputs |
| `superstore_eda.py` | Same code as a plain script (`# %%` marks notebook cells) |
| `superstore_sales.csv` | Original uploaded data (input) |
| `outputs/cleaned_superstore_sales.csv` | Cleaned dataset with engineered features (9,993 rows) |
| `outputs/figures/*.png` | 12 charts (10 required + 2 extra) |
| `outputs/Executive_Summary.pdf` | 2-page executive summary |
| `outputs/Business_Insights_and_Recommendations.md` | 7 insights (observation, evidence, meaning, action) + recommendations |

## How to run
1. Install Python 3.9+ and the libraries:
   ```
   pip install pandas numpy matplotlib seaborn reportlab jupyter
   ```
2. Keep `superstore_sales.csv` in the same folder as the notebook / script.
3. Run **either**:
   - Notebook: `jupyter notebook Superstore_EDA.ipynb`, then *Run All*, **or**
   - Script: `python superstore_eda.py`
4. Everything is (re)generated into the `outputs/` folder. `reportlab` is only needed for the PDF; if missing, that step is skipped with a message.

## Method in one minute
1. **Inspect** the raw file (shape, types, missing values, duplicates, unique values).
2. **Clean**: separate stray rows, fix data types and dates, handle missing values, remove duplicates, check categories, investigate outliers.
3. **Engineer features**: Year, Month, Year-Month, Quarter, Day, Weekday, Profit Margin, Discount Group, Sales Category (plus Shipping Days, loss and returned flags).
4. **Analyse**: sales, customers, categories, regions, discounts, seasonality.
5. **Visualise** and write business insights.

## Limitations
- Discount and profit are *associated*; the analysis does not prove discounts *cause* the losses (product mix and region also differ).
- No cost, return-reason or refund columns; data covers 2015-2018 only.
- The returns flag assumes every Order ID in the appended Returns table is a returned order.
