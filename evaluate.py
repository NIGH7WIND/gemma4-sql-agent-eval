import concurrent.futures
import functools
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

# -------------------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------------------
DATA_DIR = Path("data")
DEV_JSON = DATA_DIR / "dev.json"
DB_DIR = DATA_DIR / "database"
OUTPUT_REPORT = Path("eval_results.json")

# 64 tokens ensures the agent has room for the tool call or "TASK_COMPLETE"
model = init_chat_model(
    model="gemma4",
    model_provider="openai",
    api_key="dummy",
    base_url="http://localhost:8080/v1",
    max_tokens=300,
)

# -------------------------------------------------------------------------
# DDL Extraction & Schema Caching
# -------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def get_database_schema(db_path_str: str) -> str:
    try:
        conn = sqlite3.connect(f"file:{db_path_str}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        ddls = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return "\n\n".join(ddls)
    except Exception as e:
        return f"-- Failed to extract schema: {e}"


def build_system_prompt(schema_ddl: str) -> str:
    return f"""You are an expert SQLite Text-to-SQL assistant. You have access to the `sql_query` tool to execute and verify your query.

DATABASE SCHEMA:
{schema_ddl}

CRITICAL RULES:
1. SCHEMA FIDELITY:
   - Use strictly the table and column names defined in the schema above.
   - Do NOT invent or assume foreign keys; follow explicit schema relationships.

2. VERBATIM PROJECTION ORDER:
   - SELECT columns in the exact sequence requested in the prompt.
   - For grouped aggregations, if the aggregate is requested before the group entity (e.g., "Find average weight for each pet type"), project the aggregate first: `SELECT AVG(weight), pettype...`.

3. CASE-INSENSITIVE TEXT FILTERS:
   - SQLite text matches are case-sensitive. Always write `LOWER(col) = 'value'` for literal equality filters.

4. CLEAN NUMERIC CASTING:
   - When ordering on string numeric columns (`ORDER BY CAST(col AS REAL) ASC`), guard against non-numeric or missing data: `WHERE CAST(col AS REAL) > 0`.

5. MINIMAL TOOL USAGE (LOOP BREAKER):
   - Formulate candidate query directly from schema above.
   - Use `sql_query` to verify results. If an error occurs, correct the query immediately. Do not repeat failed queries.

6. FAST COMPLETION:
   - Once you have executed the final SQL query using `sql_query` and verified the output, DO NOT summarize or transcribe the rows.
   - Simply respond with the exact text: "TASK_COMPLETE".
"""

# -------------------------------------------------------------------------
# Timed SQLite Executor
# -------------------------------------------------------------------------
def run_sql_timed(
    db_path: Path, query: str, timeout: float = 10.0
) -> Tuple[bool, Optional[List[Tuple]], Optional[str], float]:
    conn = None
    result: Dict[str, Any] = {"ok": False, "rows": None, "err": None, "ms": 0.0}
    timed_out = threading.Event()

    def _worker():
        nonlocal conn
        t0 = time.perf_counter()
        try:
            conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)

            def _progress_handler():
                return 1 if timed_out.is_set() else 0

            conn.set_progress_handler(_progress_handler, 1000)

            cursor = conn.cursor()
            cursor.execute(query.strip().rstrip(";"))
            rows = cursor.fetchmany(1000)

            normalized = [
                tuple(round(val, 2) if isinstance(val, float) else val for val in row)
                for row in rows
            ]
            result["ok"] = True
            result["rows"] = normalized
        except Exception as e:
            result["err"] = str(e)
        finally:
            result["ms"] = (time.perf_counter() - t0) * 1000.0
            if conn:
                conn.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        timed_out.set()
        thread.join(timeout=2)
        return False, None, f"Timed out after {timeout}s", timeout * 1000.0

    return result["ok"], result["rows"], result["err"], result["ms"]


def extract_queries_from_messages(messages: List[Any]) -> Tuple[List[str], int]:
    queries = []
    tool_turns = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_turns += 1
                if call.get("name") == "sql_query":
                    q = call.get("args", {}).get("query")
                    if q:
                        queries.append(q)
    return queries, tool_turns

import itertools
from typing import List, Optional, Tuple


def compare_execution_results(
    gen_rows: Optional[List[Tuple]], gt_rows: Optional[List[Tuple]]
) -> Tuple[bool, str]:
  """Compares SQL results supporting Exact Match, Set Match, and Column-Permuted Set Match."""
  if gen_rows is None or gt_rows is None:
    return False, "Null result"

  # 1. Exact match (same order, same column alignment)
  if gen_rows == gt_rows:
    return True, "Exact Order Match"

  # 2. Row count and column count must be equal
  if len(gen_rows) != len(gt_rows):
    return False, "Row count mismatch"

  if len(gen_rows) == 0:
    return True, "Both Empty"

  if len(gen_rows[0]) != len(gt_rows[0]):
    return False, "Column count mismatch"

  # 3. Standard Row-Set Match (same columns, different row ordering)
  try:
    if set(gen_rows) == set(gt_rows):
      return True, "Set Match"
  except TypeError:
    if sorted(str(r) for r in gen_rows) == sorted(str(r) for r in gt_rows):
      return True, "Sorted Set Match"

  # 4. Permuted Column Match (handles inverted column projections)
  num_cols = len(gt_rows[0])
  if 1 < num_cols <= 5:  # Avoid factorial blow-up on large projections
    try:
      gt_set = set(gt_rows)
      for perm in itertools.permutations(range(num_cols)):
        if perm == tuple(range(num_cols)):
          continue  # Skip default ordering already tested
        # Re-index columns according to permutation
        permuted_gen = {tuple(row[i] for i in perm) for row in gen_rows}
        if permuted_gen == gt_set:
          return True, f"Permuted Column Set Match ({perm})"
    except TypeError:
      gt_sorted = sorted(str(r) for r in gt_rows)
      for perm in itertools.permutations(range(num_cols)):
        permuted_gen = sorted(
            str(tuple(row[i] for i in perm)) for row in gen_rows
        )
        if permuted_gen == gt_sorted:
          return True, f"Permuted Column Sorted Match ({perm})"

  return False, "Result Mismatch"


# -------------------------------------------------------------------------
# Thread-Isolated Evaluation Worker
# -------------------------------------------------------------------------
def process_single_test(test_tuple: Tuple[int, Dict[str, Any]], max_retries: int = 1) -> Dict[str, Any]:
    idx, test = test_tuple
    db_id = test["db_id"]
    prompt = test["question"]
    gt_sql = test["query"]
    db_path = DB_DIR / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        return {"idx": idx, "db_id": db_id, "prompt": prompt, "passed": False, "status": "DB_NOT_FOUND"}

    schema_ddl = get_database_schema(db_path.as_posix())
    system_prompt = build_system_prompt(schema_ddl)

    # Scoped directly inside the worker thread
    @tool
    def sql_query(query: str) -> str:
        """Execute a SQL query against the database to inspect or verify output"""
        ok, rows, err, _ = run_sql_timed(db_path, query, timeout=8.0)
        if not ok:
            return f"Error: {err}"
        if not rows:
            return "Success: 0 rows returned."
        # Truncate to first 3 rows to keep input context small
        return f"Success ({len(rows)} rows). Preview: {rows[:3]}"

    agent = create_agent(model=model, tools=[sql_query], system_prompt=system_prompt)

    final_query = None
    agent_latency = 0.0
    tool_turns = 0

    for attempt in range(max_retries + 1):
        t_start = time.perf_counter()
        try:
            res = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 6},
            )
            attempt_latency = time.perf_counter() - t_start
            agent_latency += attempt_latency
            messages = res.get("messages", [])

            executed_queries, turns = extract_queries_from_messages(messages)
            tool_turns += turns
            data_queries = [
                q for q in executed_queries
                if not q.strip().upper().startswith("PRAGMA")
            ]
            final_query = data_queries[-1] if data_queries else (executed_queries[-1] if executed_queries else None)

            if final_query:
                break
            elif attempt < max_retries:
                time.sleep(0.5)

        except Exception as e:
            attempt_latency = time.perf_counter() - t_start
            agent_latency += attempt_latency
            if attempt < max_retries:
                time.sleep(1.0)

    if not final_query:
        return {
            "idx": idx, "db_id": db_id, "prompt": prompt, "passed": False,
            "agent_sec": round(agent_latency, 2), "tool_turns": tool_turns, "status": "NO_SQL"
        }

    # Execute and verify
    gt_ok, gt_rows, gt_err, gt_ms = run_sql_timed(db_path, gt_sql)
    gen_ok, gen_rows, gen_err, gen_ms = run_sql_timed(db_path, final_query)

    if not gen_ok:
        return {
            "idx": idx, "db_id": db_id, "prompt": prompt, "passed": False,
            "gt_sql": gt_sql, "pred_sql": final_query, "agent_sec": round(agent_latency, 2),
            "tool_turns": tool_turns, "sql_ms": round(gen_ms, 2), "gt_ms": round(gt_ms, 2),
            "error": gen_err
        }

    eff_ratio = round(gen_ms / max(gt_ms, 0.01), 2) if gt_ok else None
    passed, match_reason = compare_execution_results(gen_rows, gt_rows)

    return {
        "idx": idx,
        "db_id": db_id,
        "prompt": prompt,
        "passed": passed,
        "match_reason": match_reason,
        "agent_sec": round(agent_latency, 2),
        "tool_turns": tool_turns,
        "sql_ms": round(gen_ms, 2),
        "gt_ms": round(gt_ms, 2),
        "efficiency_ratio": eff_ratio,
        "gt_sql": gt_sql,
        "pred_sql": final_query,
    }

# -------------------------------------------------------------------------
# Parallel Suite
# -------------------------------------------------------------------------
def evaluate_agent_parallel(sample_size: Optional[int] = 1034, max_workers: int = 4):
    if not DEV_JSON.exists() or not DB_DIR.exists():
        raise FileNotFoundError("Missing Spider files. Verify data/dev.json and data/database exist.")

    with open(DEV_JSON, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    eval_set = dev_data[:sample_size] if sample_size else dev_data
    total_samples = len(eval_set)
    test_items = list(enumerate(eval_set))

    print(f"\n{'='*75}")
    print(f"BENCHMARK RUN: {total_samples} Questions ({max_workers} Concurrent Workers)")
    print(f"{'='*75}\n")

    results_log = []
    passed_tests = 0
    total_sql_exec_time = 0.0
    bench_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_test, item): item[0] for item in test_items}

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            results_log.append(res)

            if res.get("passed"):
                passed_tests += 1

            total_sql_exec_time += res.get("sql_ms", 0.0)
            status = "PASS" if res.get("passed") else "FAIL"
            print(
                f"[{count:4d}/{total_samples}] Test #{res['idx']:<4d} "
                f"| {status:<4} | Latency: {res.get('agent_sec', 0.0):.2f}s "
                f"| DB: {res.get('db_id', 'unknown')}"
            )

    # Sort back to original dataset ordering before saving
    results_log.sort(key=lambda x: x["idx"])

    total_bench_time = time.perf_counter() - bench_start
    acc = (passed_tests / total_samples) * 100 if total_samples else 0
    avg_sql_ms = total_sql_exec_time / total_samples if total_samples else 0
    throughput = total_samples / total_bench_time if total_bench_time else 0

    print(f"\n{'='*75}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*75}")
    print(f"Questions Evaluated:         {total_samples}")
    print(f"Execution Accuracy:          {acc:.2f}% ({passed_tests}/{total_samples})")
    print(f"Total Wall-Clock Time:       {total_bench_time:.1f}s (~{total_bench_time / 60:.1f} min)")
    print(f"Throughput:                  {throughput:.2f} queries/sec")
    print(f"Avg SQLite Execution Time:   {avg_sql_ms:.2f}ms / query")
    print(f"{'='*75}\n")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "accuracy": acc,
                    "total_samples": total_samples,
                    "wall_clock_sec": round(total_bench_time, 2),
                    "throughput_qps": round(throughput, 2),
                    "avg_sql_ms": round(avg_sql_ms, 2),
                },
                "results": results_log,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    evaluate_agent_parallel(sample_size=20, max_workers=2)