from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag import vector_store


def test_streamlit_application_renders_all_tabs_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Dashboard",
        "Upload Statement",
        "Transaction Manager",
        "Spending Analytics",
        "AI Finance Assistant",
        "Reports",
    ]
    assert len(app.metric) == 11
    assert len(app.get("file_uploader")) == 1
    assert len(app.get("download_button")) == 1

    data_dir = Path(__file__).resolve().parents[1] / "data"
    app.get("file_uploader")[0].set_value(
        ("uploaded.csv", (data_dir / "sample_transactions.csv").read_bytes(), "text/csv")
    ).run()
    assert not app.exception
    assert len(app.session_state["transactions"]) == 20

    app.get("file_uploader")[0].set_value(
        ("uploaded.pdf", (data_dir / "sample_statement.pdf").read_bytes(), "application/pdf")
    ).run()
    assert not app.exception
    assert len(app.session_state["transactions"]) == 4
    assert app.session_state["transactions"]["is_valid"].all()

    app.get("selectbox")[1].set_value("Food")
    category_button = next(
        button for button in app.button if button.key == "save_transaction_category"
    )
    category_button.click().run()
    assert not app.exception
    assert app.session_state["transactions"].iloc[0]["category"] == "Food"

    uploaded_statement = (
        "Date,Narration,Amount,Type,Category\n"
        "2026-11-13,Salary deposit,5000,CR,Salary\n"
        "2026-11-14,New grocery merchant,-1250,DR,Food\n"
    ).encode("utf-8")
    app.get("file_uploader")[0].set_value(
        ("new_statement.csv", uploaded_statement, "text/csv")
    ).run()
    assert not app.exception
    assert app.session_state["data_source"] == "new_statement.csv"
    assert app.metric[0].value == "₹5,000.00"
    assert app.metric[1].value == "₹1,250.00"
    assert app.metric[3].value == "2"
    assert "New grocery merchant" in app.dataframe[1].value["description"].tolist()
    assert "New grocery merchant" in app.dataframe[2].value["description"].tolist()
    assert "Food" in app.dataframe[3].value["category"].tolist()
    manager = app.data_editor[0].value
    assert manager.loc[0, "date"] == "13-Nov-2026"
    assert manager.loc[0, "amount"] == "₹5,000.00"
    upload_metrics = {metric.label: metric.value for metric in app.metric[5:]}
    assert upload_metrics["Total imported"] == "2"
    assert upload_metrics["Total debit"] == "₹1,250.00"
    assert upload_metrics["Total credit"] == "₹5,000.00"

    app.get("chat_input")[0].set_value("How much did I spend?").run()
    assert not app.exception
    assistant_response = "\n".join(element.value for element in app.markdown)
    assert "Rule-based response" in assistant_response
    assert "₹1,250.00" in assistant_response
    assert "₹27048.00" not in assistant_response

    invalid_statement = b"Date,Merchant\n2026-11-15,Unrecognized format\n"
    app.get("file_uploader")[0].set_value(
        ("invalid.csv", invalid_statement, "text/csv")
    ).run()
    assert not app.exception
    assert app.session_state["data_source"] == "new_statement.csv"
    assert len(app.session_state["transactions"]) == 2
    assert app.metric[0].value == "₹5,000.00"


def test_transaction_manager_duplicate_checkbox_saves_and_survives_rerun(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=20
    ).run()
    duplicate_csv = (
        "Date,Description,Amount,Type,Reference Number\n"
        "2026-11-13,Repeated debit,-100,DR,ref-1\n"
        "2026-11-13,Repeated debit,-100,DR,ref-1\n"
    ).encode()
    app.get("file_uploader")[0].set_value(
        ("duplicates.csv", duplicate_csv, "text/csv")
    ).run()
    assert not app.exception
    assert app.session_state["transactions"]["duplicate_flag"].tolist() == [
        False,
        True,
    ]

    def edit_duplicate_flags(changes):
        editor_key = (
            f"transaction_manager_editor_{app.session_state['transaction_version']}"
        )
        app.session_state[editor_key] = {
            "edited_rows": changes,
            "added_rows": [],
            "deleted_rows": [],
        }
        app.run()
        save_button = next(
            button
            for button in app.button
            if button.key
            == f"save_transaction_changes_{app.session_state['transaction_version']}"
        )
        save_button.click().run()

    edit_duplicate_flags({0: {"duplicate_flag": True}})
    assert not app.exception
    assert app.session_state["transactions"]["duplicate_flag"].tolist() == [
        True,
        True,
    ]
    assert not app.session_state["transactions"]["is_valid"].any()

    edit_duplicate_flags(
        {
            0: {"duplicate_flag": False},
            1: {"duplicate_flag": False},
        }
    )

    transactions = app.session_state["transactions"]
    assert not app.exception
    assert transactions["duplicate_flag"].tolist() == [False, False]
    assert transactions["is_valid"].all()
    assert transactions["validation_errors"].eq("").all()
    assert len(transactions) == 2
    assert transactions.loc[1, "amount"] == 100
    assert transactions.loc[1, "date"].strftime("%Y-%m-%d") == "2026-11-13"

    app.run()
    assert not app.exception
    assert app.session_state["transactions"]["duplicate_flag"].tolist() == [
        False,
        False,
    ]


def test_transaction_manager_corrections_refresh_validation_errors(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=20
    ).run()
    invalid_csv = (
        "Date,Description,Amount,Type\n"
        ",Missing date,-10,DR\n"
        "2026-11-13,,-10,DR\n"
        "2026-11-13,Invalid amount,not-money,DR\n"
        "2026-11-13,Invalid type,10,unknown\n"
        "2026-11-13,Valid debit,-10,DR\n"
    ).encode()
    app.get("file_uploader")[0].set_value(
        ("invalid_rows.csv", invalid_csv, "text/csv")
    ).run()
    assert not app.exception
    initial = app.session_state["transactions"]
    assert "date" in initial.loc[0, "validation_errors"].lower()
    assert "description" in initial.loc[1, "validation_errors"].lower()
    assert "amount" in initial.loc[2, "validation_errors"].lower()
    assert "transaction type" in initial.loc[3, "validation_errors"].lower()
    assert initial.loc[4, "validation_errors"] == ""

    editor_key = (
        f"transaction_manager_editor_{app.session_state['transaction_version']}"
    )
    app.session_state[editor_key] = {
        "edited_rows": {
            0: {"date": "2026-11-12"},
            1: {"description": "Corrected description"},
            2: {"amount": 25},
            3: {"transaction_type": "income"},
        },
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    save_button = next(
        button
        for button in app.button
        if button.key
        == f"save_transaction_changes_{app.session_state['transaction_version']}"
    )
    save_button.click().run()

    corrected = app.session_state["transactions"]
    assert not app.exception
    assert corrected["is_valid"].all()
    assert corrected["validation_errors"].eq("").all()
    assert len(corrected) == 5
    assert corrected.loc[2, "amount"] == 25
    assert corrected.loc[3, "transaction_type"] == "income"


def test_transaction_manager_expands_long_descriptions(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=20
    ).run()
    description = "Synthetic merchant transaction with a long explanatory reference " * 3
    csv = f"Date,Description,Amount,Type\n2026-11-13,{description},-150,DR\n".encode()
    app.get("file_uploader")[0].set_value(("long.csv", csv, "text/csv")).run()

    assert not app.exception
    assert any(element.label == "Full transaction details" for element in app.expander)
    assert any(element.value == description.strip() for element in app.get("text"))
