import json
import os

# Plain vLLM — no centroid load/inject (used to collect training targets).
os.environ["VLLM_CENTROID_SCHEDULER"] = "0"
os.environ.setdefault("HF_HOME", "/mnt/g/agentcache/hf_cache")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID =  '/mnt/g/agentcache/models/Llama-3.2-1B-Instruct'
MAX_TOKENS = 2048
TEMPERATURE = 0.9


def generate_good_example():
    SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.

    Please solve the issue provided by the user. You can execute bash commands and edit files to implement the necessary changes.

    ## Recommended Workflow
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust"""

    # with open("tasks.json", "r") as f:
    #     tasks_data = json.load(f)
    # TASKS = [item["perturbed_task"] for item in tasks_data]

    TASKS = [
    # Bug fixes
    "I have a Python function that's supposed to return the factorial of a number but it returns wrong results for 0. Here's the code: def factorial(n): if n == 1: return 1 else: return n * factorial(n-1). Fix it.",
    "My binary search implementation never finds the target element even when it exists in the list. def binary_search(arr, target): low, high = 0, len(arr) while low < high: mid = (low+high)//2 if arr[mid] == target: return mid elif arr[mid] < target: low = mid else: high = mid return -1. What's wrong?",
    "This decorator is supposed to retry a function 3 times on exception but it only tries once. def retry(func): def wrapper(*args): try: return func(*args) except: return func(*args). Fix it.",
    "My CSV parser breaks on lines that have commas inside quoted fields. It's splitting on every comma. Fix the parsing logic.",
    "I have a race condition in my Python threading code. Two threads are incrementing a shared counter and the final value is always less than expected.",

    # Feature requests
    "Add a timeout parameter to this function that makes an HTTP GET request using the requests library. Right now it hangs forever if the server doesn't respond.",
    "I have a function that reads a JSON config file. Add support for environment variable overrides so that any key in the config can be overridden by an env var with the same name uppercased.",
    "Extend this basic logging setup to write to both console and a rotating file that caps at 10MB and keeps 3 backups.",
    "Add input validation to this Flask endpoint that accepts a JSON body with name, email, and age fields. Return proper 400 errors with messages for missing or invalid fields.",
    "I have a script that processes files sequentially. Parallelize it using Python's concurrent.futures so it processes up to 4 files at a time.",

    # Refactoring
    "This function is 200 lines long and does parsing, validation, and database insertion all in one. Break it into smaller functions with single responsibilities.",
    "I have 6 nearly identical functions that each handle a different file format (csv, json, xml, yaml, toml, ini). Refactor them into a single function with a format parameter.",
    "Replace all the raw SQL string concatenation in this file with parameterized queries to prevent SQL injection.",
    "This code uses a global variable to track state between function calls. Refactor it to use a class instead.",

    # Explanation / understanding
    "What is the difference between deepcopy and shallow copy in Python and when would each cause bugs in my code?",
    "My Python script is slow when processing a list of 1 million items. Walk me through how to profile it and identify the bottleneck.",
    "Explain why my generator function is only iterable once and how to make it reusable.",
    "Why does modifying a list inside a function affect the original list outside the function but doing the same with an integer doesn't?",

    # Testing
    "Write unit tests for a stack implementation that has push, pop, peek, and is_empty methods. Cover edge cases.",
    "I have a function that calls an external payment API. Write tests for it that mock the API so the tests don't make real network calls.",

    # Bug fixes (more)
    "My merge sort works on small lists but crashes with RecursionError on large inputs. def merge_sort(arr): if len(arr) <= 1: return arr mid = len(arr)//2; return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:])). Fix it without changing the algorithm.",
    "This LRU cache evicts the wrong item when I access keys in a certain order. class LRUCache: def __init__(self, cap): self.cap = cap; self.cache = {}. Fix the eviction logic.",
    "My context manager doesn't call __exit__ when an exception happens inside the with block. class MyCtx: def __enter__(self): return self; def __exit__(self): pass. What's missing?",
    "datetime.strptime works in tests but fails in production with the same format string. I think it's a timezone issue — help me fix the parsing.",
    "My defaultdict-based grouping returns empty lists for keys I never inserted. I expected KeyError for missing keys. Fix or explain and adjust the code.",
    "This regex is supposed to validate email addresses but it accepts strings like 'a@b'. r'^[\\w.-]+@[\\w.-]+\\.\\w+$'. Tighten it reasonably for production.",
    "My asyncio gather() hangs forever. I create tasks but never await them properly. Here's the snippet — fix it.",
    "pickle.loads() works locally but breaks when I load data saved on another machine. Same Python version. What should I change?",
    "My pandas groupby drops NaN keys silently. I need those groups included. Fix the aggregation pipeline.",
    "This property setter allows negative values on a dataclass field that should be non-negative. Add validation without breaking the dataclass API.",
    "My Flask app returns 500 on every POST because request.json is None even when I send JSON. Content-Type is application/json. Debug it.",
    "subprocess.run() with shell=True works in terminal but fails when called from cron. Exit code 127. Fix the invocation.",
    "My __hash__ and __eq__ are inconsistent and items duplicate in a set. class Point: def __eq__(self, o): return self.x == o.x and self.y == o.y. Fix it.",
    "json.dumps() fails on my object with datetime fields. Make it serializable without losing timezone info.",
    "My list comprehension builds the wrong structure — I get a flat list instead of pairs. [(x, y) for x in xs for y in ys if x < y] — I wanted nested grouping by x.",
    "sqlite3 cursor returns bytes for text columns on Linux but str on Mac in my CI. Normalize the behavior in my wrapper.",
    "My @lru_cache on a method caches across all instances incorrectly. Fix the caching so each instance has its own cache.",
    "This pathlib code raises FileNotFoundError on Windows but works on Linux for the same relative path. path = Path('data') / 'file.txt'. Fix cross-platform handling.",
    "My enum comparison breaks after I reload the module in a Jupyter notebook. class Status(Enum): OK = 1. Help me make comparisons stable.",
    "My requests session leaks connections and eventually hits 'Too many open files'. Fix the session usage pattern.",

    # Feature requests (more)
    "Add CLI flags to my argparse script for --dry-run and --verbose without breaking existing positional arguments.",
    "Add type hints to this module and make it pass mypy --strict without changing runtime behavior.",
    "Add retry with exponential backoff to my boto3 S3 upload function for transient errors.",
    "Add a progress bar with tqdm to my file copy loop over thousands of files.",
    "Add pydantic models for my FastAPI request/response bodies and migrate the endpoints to use them.",
    "Add caching with functools.lru_cache to this expensive pure function, but let me invalidate the cache from CLI.",
    "Add structured JSON logging to my script instead of plain print statements, with a log level env var.",
    "Add graceful shutdown to my asyncio worker that drains the queue before exiting on SIGTERM.",
    "Add pagination to this SQLAlchemy query that currently loads all rows into memory.",
    "Add a .env file loader using python-dotenv so local dev doesn't need exported shell vars.",
    "Add email validation and password strength checks to my signup form handler before hitting the database.",
    "Add rate limiting to my public API endpoint — max 100 requests per minute per API key.",
    "Add WebSocket support to my Flask app for live status updates on long-running jobs.",
    "Add CSV export to my Django admin list view for the Order model.",
    "Add a health check endpoint to my FastAPI service that verifies DB and Redis connectivity.",
    "Add automatic migration generation for this Alembic project when I change SQLAlchemy models.",
    "Add a context manager that temporarily changes os.environ for tests and restores afterward.",
    "Add support for reading gzip-compressed JSONL files in my ETL script without loading everything into RAM.",
    "Add a plugin hook system so third parties can register custom processors without editing core code.",
    "Add OpenTelemetry tracing spans around my HTTP client calls and database queries.",

    # Refactoring (more)
    "Convert this callback-heavy code to async/await using asyncio and aiohttp instead of nested def callbacks.",
    "Replace my hand-rolled singleton with a module-level instance or functools.lru_cache pattern — whichever fits.",
    "Extract duplicated try/except logging blocks into a decorator that logs and re-raises.",
    "Convert this giant if/elif chain on string types into a dict dispatch table of handler functions.",
    "Move all magic numbers and API URLs into a pydantic Settings class loaded from environment.",
    "Split this monolithic Django view into a service layer, serializer, and thin view function.",
    "Replace manual open()/close() file handling with context managers everywhere in this package.",
    "Refactor nested loops over users and orders into a single pandas merge for readability and speed.",
    "Convert my procedural data pipeline script into a Click CLI with subcommands for extract, transform, load.",
    "Eliminate duplicate pytest fixtures across test files by moving them to conftest.py.",
    "Replace inheritance-heavy hierarchy with composition for these notification sender classes.",
    "Refactor this God class with 40 methods into smaller mixins or separate collaborator objects.",
    "Use dataclasses instead of dicts for these records and update all access sites.",
    "Replace print debugging with proper logging and consistent logger names per module.",
    "Consolidate three nearly identical Celery tasks into one parameterized task.",
    "Refactor synchronous requests calls in my async FastAPI route to use httpx.AsyncClient properly.",
    "Extract configuration parsing from business logic in my main() function.",
    "Replace stringly-typed status codes with an Enum and update all comparisons.",
    "Move inline SQL in this repository class to SQLAlchemy ORM queries.",
    "Refactor this pytest module that uses time.sleep into freezegun or monkeypatched clocks.",

    # Explanation / understanding (more)
    "Why does my list.sort() return None but sorted(list) returns a new list? I keep assigning the result of .sort() by mistake.",
    "Explain the GIL and whether threading or multiprocessing is better for my CPU-bound image resizing job.",
    "What's the difference between __str__ and __repr__ and which should I implement for my custom class?",
    "Why does 'is' sometimes fail for small integers but '==' works? When should I never use 'is' for values?",
    "Explain how *args and **kwargs work in decorators that need to forward arguments to the wrapped function.",
    "What are descriptor protocols and why does @property work on classes but not always on instances the way I expect?",
    "Walk me through how importlib and circular imports caused my ModuleNotFoundError at runtime.",
    "Explain why mutable default arguments like def f(x=[]) are dangerous with examples from my code.",
    "What's the practical difference between asyncio.create_task, ensure_future, and gather?",
    "Why does pandas SettingWithCopyWarning appear and how do I fix my chained assignment properly?",
    "Explain when to use typing.Protocol vs ABC for duck-typed interfaces in Python 3.11+.",
    "What does if __name__ == '__main__' actually do and should my library modules have it?",
    "Why does my Docker container see a different PYTHONPATH than my local venv?",
    "Explain contextvars and when they're needed instead of thread-local storage in async code.",
    "What's the difference between subprocess, os.system, and shelling out — security and portability wise?",

    # Testing (more)
    "Write pytest parametrized tests for a function that normalizes phone numbers across US formats.",
    "Add integration tests for my Flask app using the test client, including auth headers.",
    "Write property-based tests with Hypothesis for a function that reverses a string and claims to be an involution.",
    "Mock datetime.now() in tests so my 'expires in 24h' logic is deterministic.",
    "Write tests for my custom context manager that acquires a file lock — include timeout behavior.",
    "Add snapshot tests for my HTML email template renderer so layout regressions are caught.",
    "Write async pytest tests for my aiohttp handler including error responses.",
    "Test that my SQLAlchemy model constraints reject invalid data at the DB level using a test database.",
    "Add coverage for exception paths in my retry decorator — exhaust retries and verify the last exception propagates.",
    "Write a fixture that spins up a temporary SQLite DB, runs migrations, and yields a session.",

    # Data / ML / notebooks
    "My sklearn Pipeline fails at predict time because the scaler wasn't fit on the same columns. Fix the column mismatch.",
    "Add train/val/test split with stratification on the label column in this pandas script.",
    "This matplotlib plot labels overlap on the x-axis. Fix the layout for 50+ categories.",
    "Convert my Jupyter notebook cells into a proper Python package with a CLI entry point.",
    "My torch DataLoader workers crash on Windows with multiprocessing errors. Fix the if __name__ guard pattern.",
    "Add early stopping and checkpointing to my PyTorch training loop.",
    "Vectorize this slow row-by-row pandas apply() that's killing performance on 2M rows.",
    "Fix data leakage: my feature engineering uses the full dataset before the train/test split.",
    "Add reproducibility — set random seeds for numpy, random, and torch in one place.",
    "Save and load my fitted sklearn model with joblib and version metadata.",

    # Web / APIs
    "My Django queryset N+1 queries the related User for every Order. Optimize with select_related or prefetch_related.",
    "Add CORS middleware to my FastAPI app for a React frontend on localhost:3000.",
    "Fix JWT validation in my middleware — expired tokens still get through.",
    "Add idempotency keys to my Stripe webhook handler so duplicate events don't double-charge.",
    "My GraphQL resolver returns None for nested fields that should batch-load. Add DataLoader pattern.",
    "Implement OAuth2 password flow login endpoint in FastAPI with passlib bcrypt hashing.",
    "Add request ID middleware that propagates X-Request-ID to all log lines.",
    "Fix my marshmallow schema that silently drops unknown fields — I want strict validation.",
    "Add OpenAPI examples to my FastAPI routes for better docs.",
    "My gunicorn workers share in-memory cache incorrectly. Move cache to Redis.",

    # DevOps / packaging / tooling
    "Fix my pyproject.toml so pip install -e . installs the console script entry point.",
    "Add pre-commit hooks for black, ruff, and mypy on this repo.",
    "My GitHub Actions workflow fails on 'No module named pytest'. Fix the CI Python setup.",
    "Add a Dockerfile for this Flask app with multi-stage build and non-root user.",
    "Pin dependencies in requirements.txt and explain how to generate a lockfile with pip-tools.",
    "Add tox.ini to run tests on Python 3.10, 3.11, and 3.12.",
    "Fix setuptools package discovery — tests are being installed as top-level packages.",
    "Add __version__ sourced from importlib.metadata in my library's __init__.py.",
    "My poetry project conflicts on transitive dependencies. Resolve the solver error.",
    "Add a Makefile target for lint, test, and format in one command.",

    # Security / reliability
    "Audit this code for shell injection via user input passed to os.system.",
    "Replace pickle with a safe serialization format for data from untrusted sources.",
    "Add secrets scanning — move hardcoded API keys to environment variables.",
    "Fix timing attack in my manual string comparison of HMAC signatures — use hmac.compare_digest.",
    "Add input sanitization before rendering user HTML in my Jinja2 templates.",
    "Implement constant-time token comparison for my API key auth middleware.",
    "Add connection pooling and statement timeouts to my psycopg2 usage.",
    "Fix my code that logs full credit card numbers — mask PANs in logs.",
    "Add circuit breaker pattern around my flaky downstream HTTP service.",
    "Validate file uploads — restrict extensions, size, and scan magic bytes not just filename.",

    # CLI / scripting / automation
    "Add subcommands to my Click CLI: ingest, report, and cleanup with shared global options.",
    "Parse inconsistent date formats in this CSV column into ISO 8601 datetimes.",
    "Watch a directory with watchdog and run my processor when new files appear.",
    "Send a Slack webhook notification when my batch job finishes or fails.",
    "Add argparse mutual exclusion between --json and --csv output formats.",
    "Schedule this script with APScheduler to run every weekday at 9am.",
    "Add dry-run mode that prints what would be deleted without deleting.",
    "Merge multiple Excel sheets into one dataframe with a source_sheet column.",
    "Download files from S3 with progress and resume on interruption.",
    "Add signal handlers so Ctrl+C saves partial progress before exit.",

    # Concurrency / performance
    "Profile why my multiprocessing Pool is slower than sequential — fix pickling overhead.",
    "Use asyncio.Semaphore to limit concurrent API calls to 10 at a time.",
    "Fix deadlock in my code that acquires lock A then lock B in one thread and the reverse in another.",
    "Replace busy-wait loop with threading.Event for worker shutdown.",
    "Use ProcessPoolExecutor for CPU-bound work and ThreadPoolExecutor for I/O — refactor my mixed pool.",
    "Add connection pooling to my psycopg2/asyncpg database access layer.",
    "Memoize expensive pure functions with diskcache across process restarts.",
    "Batch my INSERT statements into executemany instead of one commit per row.",
    "Use __slots__ on a high-volume data class to cut memory usage.",
    "Switch from regex to ahocorasick for matching thousands of keywords in log lines.",

    # Misc realistic agent tasks
    "Upgrade this codebase from Python 3.8 to 3.11 and fix deprecated typing imports.",
    "Add docstrings and a minimal README example for this internal utility package.",
    "Implement a simple plugin loader that imports modules from a plugins/ directory by name.",
    "Write a context manager that suppresses specific exceptions for cleanup code only.",
    "Add __all__ and make this package importable without exposing private modules.",
    "Fix my circular import between models.py and schemas.py by restructuring imports.",
    "Implement a tiny in-memory pub/sub EventBus for decoupling components in my app.",
    "Add a custom exception hierarchy for my domain errors instead of bare ValueError everywhere.",
    "Parse Apache combined log format into structured dicts with a regex or library.",
    "Build a minimal REPL that evals Python expressions safely without exec on user strings.",
]

    OUTPUT_DIR = "good_examples"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "vllm_good_examples_raw.jsonl")

    existing_data = []
    task_to_index = {}

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                existing_data.append(record)
                task_to_index[record.get("task")] = idx

    print(f"Loaded {len(existing_data)} existing records")

    for i, task in enumerate(TASKS):
        if task not in task_to_index:
            record = {
                "index": i,
                "task": task,
                "good_example": None,
            }
            task_to_index[task] = len(existing_data)
            existing_data.append(record)

    null_tasks = [r["task"] for r in existing_data if r.get("good_example") is None]
    print(f"Tasks with null good_example: {len(null_tasks)}")
    if not null_tasks:
        print("Nothing to generate.")
        return

    print(f"Loading model with vLLM: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    for i, task in enumerate(TASKS):
        rec_idx = task_to_index[task]
        record = existing_data[rec_idx]
        if record.get("good_example") is not None:
            continue

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        good_example = outputs[0].outputs[0].text.strip()
        record["index"] = i
        record["good_example"] = good_example
        existing_data[rec_idx] = record
        print(f"Filled null good_example for task {i + 1} of {len(TASKS)}")

        with open(output_path, "w") as out_f:
            for r in existing_data:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(existing_data)} records to {output_path}")


if __name__ == "__main__":
    generate_good_example()
