# Gemini Project Context: Analysis_orders

## Project Overview
**Analysis_orders** is a Python-based quantitative finance project designed to analyze transaction datasets, evaluate high-frequency/quant trading models, and generate interactive visualization reports. It focuses on metric calculation (PnL, Fill Rates, Entry/Exit Ranks) and benchmarking against market indices.

## Key Features
*   **Transaction Analysis:** Parses `orders.parquet` and `paired_trades_fifo.parquet` to compute execution metrics.
*   **Performance Attribution:** Fama-French 3-factor analysis and strategy vs. benchmark comparisons.
*   **Visualization:** Generates lightweight, interactive HTML dashboards using Plotly, suitable for GitHub Pages.
*   **Mark-to-Market (MtM):** Analyzes daily Net Asset Value (NAV) and capital utilization.

## Environment & Dependencies
*   **Language:** Python 3.x
*   **Key Libraries:** `pandas`, `numpy`, `plotly`, `scipy`, `statsmodels`, `baostock` (for market data), `pyarrow`.
*   **Setup:**
    ```bash
    pip install -r requirements.txt
    ```
*   **Data Requirements:**
    *   `data/orders.parquet`: Transaction details.
    *   `data/paired_trades_fifo.parquet`: Paired trade data (optional).
    *   `mtm_analysis_results/daily_nav_revised.csv`: Daily NAV data.

## Codebase Structure

### Core Analysis
*   **`src/lightweight_analysis.py`**: The central analysis engine. It defines the `LightweightAnalysis` class which:
    *   Loads data from `data/` and `benchmark_data/`.
    *   Calculates financial metrics (Sharpe, Volatility, Alpha/Beta).
    *   Generates a suite of Plotly figures.
    *   Compiles everything into `reports/visualization_analysis/index.html`.

### Utility Scripts
*   **`scripts/run_entry_exit_rank_baostock.py`**: A specialized script to calculate Entry and Exit Ranks (timing ability) using 5-minute market data from Baostock. It handles data caching in `data/cache/baostock_5min/`.

### Directories
*   **`data/`**: Raw input data (Parquet files) and caches.
*   **`benchmark_data/`**: Market index CSVs for benchmarking.
*   **`documents/`**: Detailed documentation on financial models (e.g., FF3 factors, slippage analysis).
*   **`reports/visualization_analysis/`**: The output directory for generated HTML reports.
*   **`docs/`**: Published reports for GitHub Pages.

## Common Tasks & Commands

### Run Full Analysis
Generates the comprehensive dashboard and all sub-reports.
```bash
python src/lightweight_analysis.py
```

### Analyze Entry/Exit Timing
Downloads necessary market data (via Baostock) and computes rank distributions.
```bash
# Ensure environment variables for proxy are cleared if necessary to access Baostock
python scripts/run_entry_exit_rank_baostock.py
```

## Development Conventions
*   **Style:** Follows PEP 8.
*   **Naming:** `snake_case` for functions/variables, `PascalCase` for classes.
*   **Language:** Comments and documentation are primarily in Chinese (as per `AGENTS.md`).
*   **Visualization:** Plotly is used for all charts to ensure interactivity and standalone HTML export capability.
*   **Version Control:** Output files (HTML reports) are generally tracked or copied to `docs/` for deployment.

## Documentation
Refer to the `documents/` directory for in-depth explanations of the mathematical models and analytical methodologies used in this project.
