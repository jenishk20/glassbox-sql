"""Each ladder step must add exactly one ingredient."""

import json
import sqlite3

import pytest

from glassbox.data import load_split
from glassbox.prompts import LADDER, build


@pytest.fixture
def bird(tmp_path):
    folder = tmp_path / "dev_databases" / "shop"
    (folder / "database_description").mkdir(parents=True)
    conn = sqlite3.connect(folder / "shop.sqlite")
    conn.execute("CREATE TABLE customer (id INTEGER PRIMARY KEY, `full name` TEXT, city TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER REFERENCES customer(id), total REAL)")
    conn.executemany("INSERT INTO customer VALUES (?, ?, ?)", [(1, "Ada Lovelace", "London"), (2, "Alan Turing", "Wilmslow")])
    conn.commit()
    conn.close()
    (folder / "database_description" / "customer.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "city,city,city where the customer lives,text,\n")
    (tmp_path / "dev.json").write_text(json.dumps([{
        "question_id": 0, "db_id": "shop", "question": "How many customers live in London?",
        "evidence": "live in refers to city", "SQL": "SELECT count(*) FROM customer WHERE city = 'London'",
        "difficulty": "simple"}]))
    return tmp_path


def test_ladder_adds_one_thing_per_step(bird):
    (ex,), info = load_split(bird, "dev")
    assert info["n"] == 1
    p = {name: build(name, ex, bird).user for name in LADDER}

    assert "Evidence:" not in p["p0-schema"] and "Evidence: live in refers to city" in p["p1-evidence"]
    assert "city where the customer lives" not in p["p1-evidence"]
    assert "-- city where the customer lives" in p["p2-descriptions"]
    assert "examples:" not in p["p2-descriptions"] and "'London'" in p["p3-examples"]
    assert "appear in the question" not in p["p3-examples"]
    assert "customer.city = 'London'" in p["p4-matches"]
    assert "step by step" not in p["p4-matches"] and "step by step" in p["p5-reasoning"]
    assert "`full name` TEXT" in p["p0-schema"]
    assert "FOREIGN KEY (customer_id) REFERENCES customer (id)" in p["p0-schema"]


def test_base_model_prompt_opens_the_sql_fence(bird):
    (ex,), _ = load_split(bird, "dev")
    assert build("p1-evidence", ex, bird).completion_text().endswith("```sql\n")
    with pytest.raises(ValueError):
        build("p5-reasoning", ex, bird).completion_text()


def test_v1_prompt_matches_old_format(bird):
    (ex,), _ = load_split(bird, "dev")
    p = build("v1", ex, bird)
    assert p.system.startswith("You are an expert SQL query generator")
    assert "External Knowledge:\nlive in refers to city" in p.user
    assert p.user.endswith("Return only the SQL query with no explanation.")
