import hashlib
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.analysis_agent import compute_summary
from agents.categorizer_agent import categorize_transactions
from agents.parser_agent import parse_statement
from agents.advisor_agent import generate_advice
from core.intent_router import detect_intent, route_question
from core.llm import get_llm, build_llm_response
from rag.retriever import get_relevant_context
from tools.transaction_tools import filter_transactions, update_transaction_category
from utils.formatting import (
    format_inr,
    format_statement_date,
    format_summary_display,
    format_transaction_display,
)
from utils.validators import standardize_dataframe


logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Statement-to-Insights Finance Agent",
    page_icon="💰",
    layout="wide",
)
def initialize_session_state():
    defaults = {
        "transactions": None,
        "analysis": None,
        "manual_category_updates": {},
        "data_source": None,
        "upload_filename": None,
        "upload_digest": None,
        "upload_error": None,
        "transaction_version": 0,
        "chat_history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_session_state()


def load_sample_transactions():
    sample_path = PROJECT_DIR / "data" / "sample_transactions.csv"
    return categorize_transactions(parse_statement(sample_path))


def transaction_option_label(transaction):
    description = str(transaction["description"])
    if len(description) > 72:
        description = f"{description[:69]}..."
    return (
        f"{format_statement_date(transaction['date'])} — {description} "
        f"({format_inr(transaction['amount'])})"
    )


def show_dashboard(df):
    summary = compute_summary(df)
    st.session_state["analysis"] = summary
    columns = st.columns(5)
    columns[0].metric("Total income", format_inr(summary["total_income"]))
    columns[1].metric("Total expenses", format_inr(summary["total_expense"]))
    columns[2].metric("Savings", format_inr(summary["net_savings"]))
    columns[3].metric("Transactions", summary["transaction_count"])
    columns[4].metric("Avg. spend / month", format_inr(summary["average_monthly_expense"]))

    monthly_df = pd.DataFrame(summary["monthly_spend"])
    if not monthly_df.empty:
        fig_monthly = px.bar(
            monthly_df,
            x="month",
            y="amount",
            title="Monthly spending overview",
            color_discrete_sequence=["#2E8B57"],
        )
        fig_monthly.update_yaxes(tickprefix="₹", tickformat=",.2f")
        fig_monthly.update_traces(
            hovertemplate="%{x}<br>Spend: ₹%{y:,.2f}<extra></extra>"
        )
        st.plotly_chart(
            fig_monthly,
            width="stretch",
            key="dashboard_monthly_spending_chart",
            alt="Monthly expenses from the currently selected transaction statement",
        )

    col1, col2 = st.columns(2)
    category_df = pd.DataFrame(summary["category_breakdown"])
    if not category_df.empty:
        category_df = category_df.sort_values("amount", ascending=False)
        fig_pie = px.pie(category_df, names="category", values="amount", title="Category-wise spending")
        fig_pie.update_traces(
            hovertemplate="%{label}<br>Spend: ₹%{value:,.2f}<extra></extra>"
        )
        col1.plotly_chart(
            fig_pie,
            width="stretch",
            key="dashboard_category_spending_chart",
            alt="Expense totals by category from the current statement",
        )

    income_expense_df = pd.DataFrame(
        [{"type": "income", "amount": summary["total_income"]}, {"type": "expense", "amount": summary["total_expense"]}]
    )
    fig_income = px.bar(income_expense_df, x="type", y="amount", title="Income vs expenditure")
    fig_income.update_yaxes(tickprefix="₹", tickformat=",.2f")
    fig_income.update_traces(
        hovertemplate="%{x}: ₹%{y:,.2f}<extra></extra>"
    )
    col2.plotly_chart(
        fig_income,
        width="stretch",
        key="dashboard_income_expenditure_chart",
        alt="Income compared with expenses from the current statement",
    )


def show_upload_tab():
    st.subheader("Upload a statement")
    uploaded_file = st.file_uploader(
        "Upload PDF or CSV statement",
        type=["pdf", "csv"],
        key="statement_upload",
    )
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        file_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        st.caption(f"Selected file: {uploaded_file.name}")
        if digest != st.session_state["upload_digest"]:
            st.session_state["upload_digest"] = digest
            st.session_state["upload_filename"] = uploaded_file.name
            st.session_state["upload_error"] = None
            temp_path = None
            try:
                if not file_bytes:
                    raise ValueError("The uploaded file is empty.")
                with st.spinner(f"Parsing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        temp_path = tmp.name
                        tmp.write(file_bytes)
                    parsed = parse_statement(temp_path)
                    parsed = categorize_transactions(parsed)
                st.session_state["transactions"] = parsed
                st.session_state["data_source"] = uploaded_file.name
                st.session_state["transaction_version"] += 1
                st.session_state["chat_history"] = []
                st.success(f"Loaded {len(parsed)} transaction rows from {uploaded_file.name}.")
                st.rerun()
            except ValueError as exc:
                st.session_state["upload_error"] = str(exc)
                st.error(f"Could not parse {uploaded_file.name}: {exc}")
                if st.session_state["data_source"] != uploaded_file.name:
                    st.info("The previously loaded transaction data remains active.")
            except Exception:
                logger.exception("Unexpected statement parsing failure for extension %s.", suffix)
                st.session_state["upload_error"] = (
                    "An unexpected error occurred while parsing this statement. "
                    "Check that the file is a valid, readable CSV or PDF and try again."
                )
                st.error(st.session_state["upload_error"])
                if st.session_state["data_source"] != uploaded_file.name:
                    st.info("The previously loaded transaction data remains active.")
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        logger.warning("Could not remove temporary statement upload file.")
        elif st.session_state["upload_error"]:
            st.error(
                f"Could not parse {uploaded_file.name}: {st.session_state['upload_error']}"
            )
        else:
            st.success(f"Loaded {uploaded_file.name}.")
    elif st.session_state["data_source"]:
        st.session_state["upload_digest"] = None
        st.caption(f"Current transaction source: {st.session_state['data_source']}")

    if st.session_state["transactions"] is not None:
        transactions = st.session_state["transactions"]
        valid_count = int(transactions["is_valid"].fillna(False).sum())
        invalid_count = len(transactions) - valid_count
        duplicate_count = int(transactions["duplicate_flag"].fillna(False).sum())
        valid_transactions = transactions.loc[
            transactions["is_valid"].fillna(False)
        ]
        total_debit = valid_transactions.loc[
            valid_transactions["transaction_type"].eq("expense"), "amount"
        ].sum()
        total_credit = valid_transactions.loc[
            valid_transactions["transaction_type"].eq("income"), "amount"
        ].sum()
        st.subheader("Import summary")
        summary_metrics = st.columns(6)
        summary_metrics[0].metric("Total imported", len(transactions))
        summary_metrics[1].metric("Valid", valid_count)
        summary_metrics[2].metric("Invalid", invalid_count)
        summary_metrics[3].metric("Possible duplicates", duplicate_count)
        summary_metrics[4].metric("Total debit", format_inr(total_debit))
        summary_metrics[5].metric("Total credit", format_inr(total_credit))
        st.caption(
            "Debit and credit totals include valid, unflagged transactions only. "
            "Possible duplicates remain imported for manual review."
        )
        st.subheader("Validation preview")
        st.dataframe(
            format_transaction_display(transactions.head(100)),
            width="stretch",
            key="upload_validation_preview",
            alt="Parsed transactions and validation results",
        )
        invalid_rows = transactions.loc[~transactions["is_valid"].fillna(False)]
        if not invalid_rows.empty:
            st.warning("Review rows marked invalid. Correct date, description, amount, and transaction type where possible.")
            version = st.session_state["transaction_version"]
            editor_key = f"invalid_transaction_editor_{version}"
            edit_view = invalid_rows[
                ["date", "description", "amount", "transaction_type", "category", "validation_errors"]
            ].copy()
            if "reference_number" in invalid_rows:
                edit_view["reference_number"] = invalid_rows["reference_number"]
            else:
                edit_view["reference_number"] = ""
            edit_view.insert(0, "row_number", [str(index) for index in invalid_rows.index])
            edit_view["date"] = edit_view["date"].map(format_statement_date)
            edit_view["amount"] = edit_view["amount"].map(format_inr)
            edit_view["transaction_type"] = edit_view["transaction_type"].fillna("")
            for column in ("debit", "credit"):
                if column in edit_view:
                    edit_view[column] = edit_view[column].map(format_inr)
            st.data_editor(
                edit_view,
                key=editor_key,
                hide_index=True,
                disabled=["row_number", "validation_errors"],
                column_config={
                    "transaction_type": st.column_config.SelectboxColumn(
                        "Transaction type",
                        options=["", "income", "expense"],
                    ),
                    "amount": st.column_config.TextColumn(
                        "Amount",
                        help="Enter the transaction amount; validation will report invalid values.",
                    ),
                    "date": st.column_config.TextColumn(
                        "Date",
                        help="Displayed as DD-MMM-YYYY.",
                    ),
                    "validation_errors": st.column_config.TextColumn("Why it needs review"),
                },
                width="stretch",
                num_rows="fixed",
            )
            if st.button("Apply transaction corrections", key=f"apply_transaction_corrections_{version}"):
                corrected = transactions.copy()
                for _, row in edited.iterrows():
                    original_index = row["row_number"]
                    try:
                        original_index = int(original_index)
                    except (TypeError, ValueError):
                        pass
                    if original_index not in corrected.index:
                        continue
                    for column in (
                        "date",
                        "description",
                        "amount",
                        "transaction_type",
                        "category",
                        "reference_number",
                    ):
                        if column == "reference_number" and column not in corrected:
                            corrected[column] = ""
                        corrected.at[original_index, column] = row[column]
                corrected = standardize_dataframe(corrected)
                st.session_state["transactions"] = categorize_transactions(corrected)
                st.session_state["transaction_version"] += 1
                st.success("Corrections applied. The dashboard and assistant now use the reviewed transactions.")
                st.rerun()


def show_transaction_manager(df):
    st.subheader("Transaction Manager")
    if df is None:
        st.info("Upload a statement before editing transactions.")
        return

    filtered = filter_transactions(df)
    if filtered.empty:
        st.info("No transactions to display.")
        return

    st.caption(
        "Use the Duplicate flag checkbox to mark or clear a possible duplicate. "
        "Saving keeps every transaction; flagged rows are not deleted."
    )
    manager_view = filtered[
        [
            "transaction_id",
            "date",
            "description",
            "amount",
            "transaction_type",
            "duplicate_flag",
            "validation_errors",
        ]
    ].copy()
    manager_view["date"] = manager_view["date"].map(format_statement_date)
    manager_view["amount"] = manager_view["amount"].map(format_inr)
    manager_view["transaction_type"] = (
        manager_view["transaction_type"].astype("string").replace("<NA>", "")
    )
    version = st.session_state["transaction_version"]
    editor_key = f"transaction_manager_editor_{version}"
    edited = st.data_editor(
        manager_view,
        width="stretch",
        key=editor_key,
        hide_index=True,
        num_rows="fixed",
        disabled=["transaction_id", "validation_errors"],
        column_config={
            "transaction_id": None,
            "date": st.column_config.TextColumn("Date", help="Displayed as DD-MMM-YYYY."),
            "description": st.column_config.TextColumn(
                "Description",
                width="large",
                help="Select a transaction below to expand and review its full description.",
            ),
            "amount": st.column_config.TextColumn(
                "Amount",
                help="Displayed in Indian Rupees; edit the value if a correction is needed.",
            ),
            "transaction_type": st.column_config.TextColumn("Transaction type"),
            "duplicate_flag": st.column_config.CheckboxColumn(
                "Duplicate flag",
                help=(
                    "Marks a transaction as a possible duplicate. This is a review flag, "
                    "not a deletion; clear it if the transaction is legitimate."
                ),
            ),
            "validation_errors": st.column_config.TextColumn(
                "Validation error",
                help="Blank means the transaction passed validation.",
            ),
        },
        alt="Transaction list with categories and amounts",
    )
    if st.button("Save transaction changes", key=f"save_transaction_changes_{version}"):
        updated = df.set_index("transaction_id", drop=False).copy()
        editor_state = st.session_state.get(editor_key, {})
        edited_rows = editor_state.get("edited_rows", {})
        if "duplicate_flag_override" not in updated:
            updated["duplicate_flag_override"] = pd.Series(
                pd.NA, index=updated.index, dtype="boolean"
            )
        for row_position, changes in edited_rows.items():
            try:
                transaction_id = filtered.iloc[int(row_position)]["transaction_id"]
            except (IndexError, TypeError, ValueError):
                continue
            if transaction_id not in updated.index:
                continue
            for column in ("date", "description", "amount", "transaction_type"):
                if column in changes:
                    updated.at[transaction_id, column] = changes[column]
            if "duplicate_flag" in changes:
                updated.at[transaction_id, "duplicate_flag_override"] = bool(
                    changes["duplicate_flag"]
                )
        updated = standardize_dataframe(updated.reset_index(drop=True))
        st.session_state["transactions"] = updated
        st.session_state["transaction_version"] += 1
        st.success("Transaction changes saved. Validation errors were refreshed.")
        st.rerun()

    selected_index = st.selectbox(
        "Select a transaction to edit",
        list(range(len(filtered))) or [-1],
        format_func=lambda index: (
            transaction_option_label(filtered.iloc[index])
            if index >= 0
            else "No transactions"
        ),
        key="transaction_to_edit",
    )
    if selected_index >= 0:
        txn = filtered.iloc[selected_index]
        with st.expander("Full transaction details"):
            st.text(str(txn["description"]))
            if txn["validation_errors"]:
                st.warning(f"Validation: {txn['validation_errors']}")
        category_options = [
            "Food",
            "Shopping",
            "Transport",
            "Bills",
            "Entertainment",
            "Healthcare",
            "Education",
            "Rent",
            "Salary",
            "Transfers",
            "Investments",
            "Others",
        ]
        if txn["category"] not in category_options:
            category_options.append(txn["category"])
        new_category = st.selectbox(
            "Update category",
            category_options,
            index=category_options.index(txn["category"]),
            key="transaction_category_update",
        )
        if st.button("Save category update", key="save_transaction_category"):
            updated = update_transaction_category(df, txn["transaction_id"], new_category)
            st.session_state["transactions"] = updated
            st.success(f"Updated category for {txn['description']} to {new_category}.")
            st.rerun()


def show_analytics(df):
    if df is None:
        st.info("Upload financial data to view analytics.")
        return
    summary = compute_summary(df)
    category_df = pd.DataFrame(summary["category_breakdown"])

    if not category_df.empty:
        category_df = category_df.sort_values("amount", ascending=False)
        pie_chart = px.pie(category_df, names="category", values="amount", title="Category-wise spending")
        pie_chart.update_traces(
            hovertemplate="%{label}<br>Spend: ₹%{value:,.2f}<extra></extra>"
        )
        st.plotly_chart(
            pie_chart,
            width="stretch",
            key="analytics_category_spending_chart",
            alt="Expense totals by category",
        )

    monthly_df = pd.DataFrame(summary["monthly_spend"])
    if not monthly_df.empty:
        bar_chart = px.bar(monthly_df, x="month", y="amount", title="Monthly spending")
        bar_chart.update_yaxes(tickprefix="₹", tickformat=",.2f")
        bar_chart.update_traces(
            hovertemplate="%{x}<br>Spend: ₹%{y:,.2f}<extra></extra>"
        )
        st.plotly_chart(
            bar_chart,
            width="stretch",
            key="analytics_monthly_spending_chart",
            alt="Monthly expense totals",
        )

    st.dataframe(
        format_summary_display(pd.DataFrame(summary["top_expenses"])),
        width="stretch",
        key="analytics_top_expenses_table",
        alt="Five largest expenses in the selected statement",
    )


def show_ai_assistant(df):
    st.subheader("AI Finance Assistant")
    llm = get_llm()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            if answer.startswith("AI-generated response:"):
                st.caption("AI-generated, using transaction summaries and retrieved context")
            elif "Rule-based response" in answer:
                st.caption("Rule-based fallback; no AI provider response was used")
            st.write(answer)

    user_query = st.chat_input(
        "Ask about spending, categories, or savings",
        key="finance_assistant_chat_input",
    )
    if user_query:
        intent = detect_intent(user_query, llm)
        route = route_question(intent)
        summary = compute_summary(df) if df is not None else {}
        context = get_relevant_context(user_query, summary=summary)
        if df is None:
            response = "Please upload transaction data first."
        else:
            if route == "analysis":
                response = build_llm_response(
                    user_query=user_query,
                    route=route,
                    summary=summary,
                    context=context,
                    llm=llm,
                )
            elif route == "advisor":
                advice = generate_advice(df)
                response = build_llm_response(
                    user_query=user_query,
                    route=route,
                    summary={**summary, "advice": advice},
                    context=context,
                    llm=llm,
                )
            else:
                response = build_llm_response(
                    user_query=user_query,
                    route=route,
                    summary={
                        **summary,
                        "records": df[["date", "description", "amount", "category"]]
                        .head(5)
                        .to_dict(orient="records"),
                    },
                    context=context,
                    llm=llm,
                )

        st.session_state.chat_history.append((user_query, response))
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            if response.startswith("AI-generated response:"):
                st.caption("AI-generated, using transaction summaries and retrieved context")
            elif "Rule-based response" in response:
                st.caption("Rule-based fallback; no AI provider response was used")
            st.write(response)


def show_reports(df):
    st.subheader("Reports")
    if df is None:
        st.info("Upload transactions to create a report.")
        return

    summary = compute_summary(df)
    report_df = pd.DataFrame(summary["category_breakdown"])
    display_report_df = format_summary_display(report_df)
    st.dataframe(
        display_report_df,
        width="stretch",
        key="reports_category_breakdown_table",
        alt="Category spending report",
    )
    csv_buffer = BytesIO()
    display_report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download category report",
        data=csv_buffer.getvalue(),
        file_name="finance_report.csv",
        mime="text/csv",
        key="download_category_report",
    )

    monthly_report = format_summary_display(pd.DataFrame(summary["monthly_spend"]))
    st.dataframe(
        monthly_report,
        width="stretch",
        key="reports_monthly_spending_table",
        alt="Monthly spending report",
    )


def main():
    st.title("Statement-to-Insights Finance Agent")
    st.caption("Upload bank and UPI statements, query spending patterns, and get grounded financial guidance.")

    if st.session_state["transactions"] is None:
        st.session_state["transactions"] = load_sample_transactions().copy()
        st.session_state["data_source"] = "Built-in sample data"

    tabs = st.tabs([
        "Dashboard",
        "Upload Statement",
        "Transaction Manager",
        "Spending Analytics",
        "AI Finance Assistant",
        "Reports",
    ])

    with tabs[0]:
        show_dashboard(st.session_state["transactions"])
    with tabs[1]:
        show_upload_tab()
    with tabs[2]:
        show_transaction_manager(st.session_state["transactions"])
    with tabs[3]:
        show_analytics(st.session_state["transactions"])
    with tabs[4]:
        show_ai_assistant(st.session_state["transactions"])
    with tabs[5]:
        show_reports(st.session_state["transactions"])


if __name__ == "__main__":
    main()
