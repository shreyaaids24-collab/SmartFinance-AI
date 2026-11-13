# Statement-to-Insights Finance Agent

A Streamlit personal-finance demo that reads bank/UPI CSV or text-based PDF statements, validates and categorizes transactions, calculates spending summaries, and answers finance questions using retrieved guidance. It works in rule-based mode without API credentials; an optional OpenAI or local Ollama chat model can generate responses.

> **Academic demo, not financial advice.** Check extracted transactions before relying on summaries. The parser does not perform OCR, connect to bank accounts, or guarantee support for every bank's statement layout.

## Features

- Parse CSV files and text-based PDFs into normalized transactions.
- Recognize common date, narration, amount, debit, credit, and transaction-type column names.
- Flag rows with missing or malformed dates, amounts, descriptions, or transaction types for review.
- Categorize transactions using merchant keywords, with optional LLM classification.
- View income, expenses, net savings, category breakdowns, monthly expense totals, and largest expenses.
- Edit transaction categories and download a category report as CSV.
- Ask questions through an intent-routed assistant. ChromaDB retrieves bundled budgeting guidance and sanitized category/month totals.
- Use deterministic rule-based summaries and recommendations when an LLM is not configured or a request fails.

## Technology

- Python, pandas, Streamlit, and Plotly
- pdfplumber for text extraction from PDFs
- ChromaDB with deterministic local hashing embeddings for offline retrieval
- LangChain integrations for optional OpenAI and Ollama chat models
- pytest and Streamlit `AppTest`

## Project layout

```text
finance-agent/
├── app.py
├── agents/       # Parser, categorizer, analysis, and advisor
├── core/         # Configuration, prompts, model integration, and intent routing
├── data/         # Public demo CSV and PDF; ChromaDB data is created here at runtime
├── rag/          # Local embeddings, ChromaDB indexing, and retrieval
├── tools/        # Calculations, transaction operations, and data helpers
├── utils/        # CSV/PDF readers and transaction validation
├── tests/        # Parser, validation, agent, RAG, routing, and UI tests
├── .env.example  # Configuration template; contains no real credentials
├── architecture_diagram.md
├── requirements.txt
└── README.md
```

See [architecture_diagram.md](architecture_diagram.md) for the workflow diagram.

## Setup

Use Python 3.10 or newer. From PowerShell, in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Optional AI provider

The app runs without a provider/API key and labels assistant output as **Rule-based**. For OpenAI:

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` to your own key and leave `LLM_PROVIDER=openai`.

For a local Ollama installation, set `LLM_PROVIDER=ollama`, configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, and make sure the model is available locally. Never commit `.env` or share its contents.

When an external chat provider is enabled, the assistant sends the user's question and the relevant transaction summary/context to that provider. Avoid uploading statements containing information you do not want processed by the configured provider. ChromaDB stores bundled guidance and aggregate category/month summaries, not raw transaction descriptions.

## Run

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The sample CSV is loaded on first launch. Use **Upload Statement** to select a `.csv` or `.pdf` file.

### Statement upload formats and limitations

CSV uploads need a transaction date, a description/narration, and either an amount or debit/credit columns. The reader detects comma, semicolon, tab, and pipe delimiters; supports UTF-8 (including BOM), Windows-1252, and Latin-1 text; and preserves quoted descriptions containing delimiters. Common headers include `Date`/`Transaction Date`/`Value Date`, `Description`/`Narration`/`Particulars`, `Amount`, `Debit`/`Withdrawal`, `Credit`/`Deposit`, `Type`, and reference-number variants. Dates include `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`, and `YYYY-MM-DD`. Indian grouped amounts (for example `₹12,34,567.50`), currency symbols, signs, and `CR`/`DR` suffixes are recognized.

Text-based PDFs can be parsed when the statement has a detectable date, description, debit/withdrawal, and credit/deposit layout. The reader supports extracted tables, delimited or aligned text, and position-aligned rows whose descriptions continue onto following lines. It handles multiple pages and common date/amount formats above. The supplied statement layout was tested. Scanned/image-only PDFs require OCR, which is not included. Arbitrary or encrypted PDF layouts are not guaranteed; remove a PDF password before upload.

Transactions with a separate nonzero debit or credit are assigned direction from that column. Signed amounts and explicit `CR`/`DR` or recognized transaction-type values also provide direction. A positive unsigned amount with no direction evidence is flagged for review rather than guessed. Invalid rows remain visible for correction and are excluded from analysis; duplicate rows are flagged.

To upload, open **Upload Statement**, choose a `.csv` or `.pdf`, check the filename and validation preview, and correct any flagged rows before using the dashboard and reports. A successfully parsed statement replaces the active dataset; if parsing fails, the previously loaded data remains active. All tabs, including the assistant's current-statement context, use the active dataset.

In **Transaction Manager**, edit transaction fields or toggle **Duplicate flag**, then select **Save transaction changes**. The flag marks a possible duplicate for review; it never deletes a row. Saving recalculates validation errors, which are blank for valid transactions. Transaction data and edits remain available across reruns in the active Streamlit session; a full browser reload or new session resets to the built-in sample because uploaded statements are not persisted to disk.

Dates in transaction tables are displayed as `DD-MMM-YYYY`; amounts use Indian Rupee formatting with Indian comma grouping and two decimal places. Upload summary totals include imported, valid, invalid, and possible duplicate row counts plus debit/credit totals for valid, unflagged transactions. Category inference uses clear merchant/description matches only; unclear or conflicting descriptions remain `Others` for review.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite covers CSV/PDF parsing, validation, categorization, analysis, advice, intent routing, local ChromaDB retrieval, rule-based assistant responses, and a Streamlit screen/transaction-manager smoke test.

## Screenshots

The `screenshots/` directory contains captures made from the running app using the included sample data:

- `screenshots/dashboard.png` — sample income/expense metrics and dashboard.
- `screenshots/assistant-rule-based.png` — assistant response without an API key, including retrieved context.

If adding more screenshots, use only public sample data (or synthetic data you are permitted to share).

## Submission hygiene

- Include source, tests, documentation, `.env.example`, and the two sample statement files.
- Exclude `.venv/`, `.env`, `data/chroma/`, caches, temporary uploads, and private statements.
- Create the ZIP from the `finance-agent` project directory after tests pass.
