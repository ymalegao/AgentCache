import json
import os
import argparse
from pathlib import Path

# Plain vLLM, no centroid load or inject. Used to collect training targets.
os.environ["VLLM_CENTROID_SCHEDULER"] = "0"
os.environ.setdefault("HF_HOME", "/mnt/g/agentcache/hf_cache")

DEFAULT_MODEL_ID = '/mnt/g/agentcache/models/qwen-1.5b'
DEFAULT_SYSTEM_PROMPT = "agentcache_compression/prompts/2000_search_agent_system.txt"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.9


def generate_good_example():
    ap = argparse.ArgumentParser(description="Generate good_examples/vllm_good_examples_raw.jsonl via vLLM.")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model path or HF id.")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt .txt path.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens per example.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    ap.add_argument(
        "--ensure-goodbye",
        action="store_true",
        help="Append '\\nGOODBYE' only if the model output didn't end with exact token GOODBYE.",
    )
    args = ap.parse_args()

    system_prompt_path = Path(args.system_prompt)
    system_prompt = system_prompt_path.read_text().strip()

    # Heavy deps: import after arg parsing so `--help` works without the venv.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # with open("tasks.json", "r") as f:
    #     tasks_data = json.load(f)
    # TASKS = [item["perturbed_task"] for item in tasks_data]

    PYTHON_TASKS = [
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


    TASKS = [
    # Fact-checking & Verification
    "Verify the claim that a new room-temperature superconductor was replicated in 2025. Find the original paper and the consensus among major labs.",
    "Is it true that the standard library of Go suffered a major supply chain vulnerability last year? Find the CVE number and patch details.",
    "Check if the reported 2026 inflation rate for the Eurozone matches the European Central Bank's initial forecasts. Provide the exact figures.",
    "An article claims that Python 3.13 completely removed the GIL by default. Verify if this is true or if it requires a specific build flag.",
    "Look up the latest legal rulings on AI-generated art copyright from this year. Is training on copyrighted data still considered fair use in the US?",

    # Technical Deep Dives
    "What are the main architectural differences between transformers and Mamba SSMs?",
    "Explain how RLHF differs from DPO in LLM fine-tuning.",
    "What is the current state of quantum computing for practical applications?",
    "Summarize the key ideas behind retrieval-augmented generation.",
    "How does PagedAttention improve GPU memory efficiency in LLM serving?",

    # Academic & Paper Summaries
    "What were the main contributions of the Attention is All You Need paper?",
    "Find the paper 'Direct Preference Optimization' and summarize its core methodology and how it avoids a reward model.",
    "Locate the original FlashAttention paper. What hardware bottleneck did the authors target, and what was their exact speedup?",
    "Summarize the key findings of the Chinchilla scaling laws paper regarding optimal token-to-parameter ratios.",

    # Concept Explanations
    "Explain the difference between KV cache quantization and weight quantization.",
    "What is speculative decoding and how does it speed up inference?",
    "How does FlashAttention reduce memory usage compared to standard attention?",
    "What are the tradeoffs between beam search and sampling for text generation?",
    "Explain how LoRA fine-tuning works and why it is parameter-efficient.",

    # Synthesis & Comparison
    "Compare the performance and cost of pgvector vs. dedicated vector databases like Pinecone and Milvus for production RAG.",
    "What is the difference between pre-training and instruction tuning?",

    # Fact-checking & Verification (More)
    "Verify whether the James Webb Space Telescope actually discovered direct atmospheric evidence of life on an exoplanet recently.",
    "Look up the definitive specs of the latest NVIDIA Blackwell architecture. What is the real-world FP4 compute performance?",
    "Is the 'xz utils' backdoor from 2024 still affecting any active Linux LTS distributions, or has it been completely mitigated?",
    "Check the latest status of the Apache Software Foundation's license dispute with Elastic. Did they reach a new agreement?",
    "Verify the maximum context window size of the latest Gemini 1.5 Pro model as of early 2026. Is it still 2 million tokens?",
    "Find out if any country has successfully banned algorithmic high-frequency trading in equity markets as of this year.",
    "Look up the current GitHub star rankings for the top 5 Python web frameworks. Who is leading between FastAPI and Django?",
    "Verify the claim that OpenAI's Sora was trained entirely on synthetic data generated by Unreal Engine 5.",
    "Find the specific RFC number that defines the HTTP/3 protocol and check if there are any new updates or errata filed recently.",
    "Check if the Python Steering Council has officially accepted the PEP for adding a JIT compiler to the core language.",
    "Verify whether Apple has allowed third-party browser engines on iOS outside of the European Union as of 2026.",
    "Find the latest benchmarks comparing Bun and Node.js for HTTP server throughput. Which one holds the current crown?",
    "Is it true that the Rust programming language foundation changed its trademark policy this year? Find the community reaction.",
    "Check the current market share statistics for cloud providers (AWS, Azure, GCP). Did Azure close the gap in 2025?",
    "Find the exact date and details of the upcoming Ethereum network upgrade. What are the primary EIPs included?",
    "Verify whether the US government has passed any federal data privacy laws targeting AI companies' scraping habits.",
    "Look up the latest CVEs for the Redis database. Are there any unpatched remote code execution vulnerabilities active?",
    "Check if Meta has open-sourced the weights for Llama 4 yet, or what their official timeline announcement says.",
    "Find the real-world power consumption metrics for running a 70B parameter model inference vs. a standard web server.",
    "Verify if Microsoft has completely deprecated the standard Windows Command Prompt in favor of Terminal by default.",

    # Market Research & Tech Trends
    "What are the trending open-source alternatives to Jira for agile project management in 2026?",
    "Provide a detailed market analysis of the shift from REST APIs to gRPC in microservices architecture over the last 3 years.",
    "What are the leading tools for tracking cloud carbon footprint emissions across AWS and Google Cloud?",
    "Investigate the adoption rate of Rust within the Linux kernel development community over the past 24 months.",
    "What are the current industry standard pricing models for enterprise LLM API access (per million tokens)?",
    "Find the top-rated developer tools for monitoring and debugging asynchronous Python code in production.",
    "What is the current state of developer adoption for WebAssembly (Wasm) in serverless edge computing?",
    "Analyze the growth of vector databases in the enterprise. Which provider secured the most market funding last year?",
    "What are the primary security tools teams are using to scan for vulnerabilities in Infrastructure as Code (IaC)?",
    "Identify the major tech companies that transitioned from a remote-first policy back to strict RTO in 2025/2026.",
    "What are the most popular open-source frameworks for building multi-agent AI workflows right now?",
    "Analyze the developer sentiment regarding the transition from Webpack to Vite in large-scale frontend applications.",
    "What is the current market penetration of alternative database engines like DuckDB for local analytics?",
    "Provide a breakdown of the current top-paying programming languages according to the latest Stack Overflow developer survey.",
    "What are the most significant compliance hurdles for healthcare startups adopting cloud-hosted LLMs under HIPAA?",
    "Investigate the rise of 'Green Computing' initiatives. Which major cloud provider has the highest renewable energy percentage?",
    "What are the top deployment platforms for hosting independent, open-source AI models without heavy infrastructure setup?",
    "Find industry reports on how generative AI coding assistants have impacted developer velocity and code quality.",
    "What is the current adoption rate of GraphQL vs REST in newly launched public developer APIs?",
    "Identify the leading frameworks for cross-platform desktop application development, comparing Electron, Tauri, and Flutter.",

    # Troubleshooting & Knowledge Retrieval
    "Find the documented solution for the OpenSSL 'unsafe legacy renegotiation disabled' error when connecting to older servers.",
    "What is the recommended fix for a Docker container failing with 'No space left on device' when df shows plenty of disk space?",
    "Search for known workarounds for the PostgreSQL 'could not serialize access due to concurrent update' isolation error.",
    "What causes the 'headers already sent' error in Node.js express applications and what is the definitive pattern to prevent it?",
    "Look up the standard resolution for Python's 'ValueError: source code string cannot contain null bytes' during script execution.",
    "Find the GitHub issue discussing the memory leak in the requests library when using deep copied sessions, and provide the fix.",
    "What is the standard way to debug an unexpected 'SIGKILL' error on a Kubernetes pod running a Python data processing script?",
    "Search for the solution to Git's 'fatal: refusing to merge unrelated histories' when combining two independent repositories.",
    "What is the recommended configuration to prevent Nginx from dropping long-lived WebSocket connections after 60 seconds?",
    "Find out why a PyTorch model might throw an 'unbound local variable' error specifically during distributed data parallel (DDP) training.",
    "What are the common causes of a 'Connection pool exhausted' error in SQLAlchemy when using Celery workers?",
    "Look up why the 'SettingWithCopyWarning' in pandas occurs even when using `.loc` in nested conditional statements.",
    "Search for the definitive guide to resolving circular import issues in modern FastAPI applications using APIRouter.",
    "What causes an AWS Lambda function to fail with 'Task timed out after X seconds' when processing S3 events sequentially?",
    "Find the fix for the React 18 'hydration mismatch' error when rendering server-side components with dynamic timestamps.",
    "Why does sqlite3 throw a 'database is locked' error in a multi-threaded Python application and how do you change the timeout?",
    "What is the standard resolution for a 'Permission denied (publickey)' error when pushing to GitHub via a new SSH key?",
    "Search for the known bug in the python-dotenv package regarding parsing variables that contain unquoted hashtags.",
    "Why does an async loop fail with 'RuntimeError: Event loop is closed' when shutting down a closing aiohttp client session?",
    "Find the recommended way to handle timezone-naive datetimes causing parsing errors in Pydantic v2 models.",

    # Explanation / Understanding (More)
    "Explain the scaling laws for LLMs and what they predict.",
    "What is mixture of experts and how does it affect model capacity vs compute?",
    "What is chain-of-thought prompting and when does it help most?",
    "Explain how vector databases work for semantic search.",
    "What are the main failure modes of RAG systems in production?",
    "How does temperature affect LLM output diversity and quality?",
    "Explain how multi-head attention differs from single-head attention.",
    "What is the purpose of the KV cache in autoregressive generation?",
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

    # Standards, Policy, & Legal Research
    "Summarize the compliance requirements for the EU AI Act regarding high-risk foundational models.",
    "What are the current SOC2 Type II audit requirements regarding data encryption for cloud-native SaaS startups?",
    "Find the differences between the MIT license and the Apache 2.0 license regarding patent rights and redistribution.",
    "What are the legal implications of the latest FTC ruling on automatic subscription renewals for digital software?",
    "Summarize the GDPR guidelines regarding the 'right to be forgotten' when a user's data has been used in an LLM training set.",
    "What is the current legal status of web scraping for public data in the US following recent appeals court rulings?",
    "What are the technical controls required to meet PCI-DSS 4.0 compliance for a custom cloud payment gateway?",
    "Find the official guidelines on how federal agencies must assess software supply chain security using SBOMs.",
    "What are the copyright risks associated with using open-source AI code assistants trained on GPL-licensed repositories?",
    "Summarize the ISO/IEC 42001 standard for Artificial Intelligence Management Systems.",

    # Historical Tech Reference
    "Trace the evolution of the Python package manager from easy_install, to pip, to modern tools like Poetry anduv.",
    "What were the technical reasons behind the failure of the Yahoo search engine dominance in the early 2000s?",
    "Provide a historical timeline of the browser wars, detailing how Google Chrome overtook Internet Explorer.",
    "What was the 'Heartbleed' vulnerability, how did it work, and what impact did it have on global internet security?",
    "Explain the history of the NoSQL movement. What triggered the massive shift toward MongoDB and Cassandra in the late 2000s?",
    "What were the core architectural flaws that led to the deprecation of Python 2, and why did the migration to Python 3 take a decade?",
    "Trace the origins of Kubernetes back to Google's internal Borg system. What features were brought over directly?",
    "What caused the dot-com crash of 2000? Provide a technical and financial synthesis of the infrastructure overvaluation.",
    "Explain the history of the Unix epoch timestamp. Why was January 1st, 1970 chosen, and what happens in the year 2038?",
    "How did the concept of MapReduce revolutionize big data processing, and why did Apache Spark eventually replace it?",

    # Multi-Step Research & Synthesis
    "Research the top 3 vulnerabilities in modern smart contracts. Synthesize their mechanics, real-world exploits, and mitigations.",
    "Compare the developer experience, performance, and operational cost of AWS Lambda vs Cloudflare Workers for edge APIs.",
    "Investigate the consensus on using Monorepos vs Polyrepos in 2026. Highlight the tools used by big tech to manage monorepos.",
    "Analyze the security landscape of container escape vulnerabilities. Explain the top 3 techniques used to break out of Docker.",
    "Provide a complete comparative study of the leading front-end state management libraries across React, Vue, and Svelte.",
    "Research how top-tier engineering teams handle database migrations with zero downtime for multi-terabyte PostgreSQL databases.",
    "Synthesize the various methods used to evaluate LLM hallucination rates. Compare benchmarks like TruthfulQA and HaluEval.",
    "Investigate the latency and throughput implications of using gRPC-Web vs traditional REST for real-time dashboard UI data.",
    "Provide a landscape review of open-source model optimization toolkits. Compare TensorRT-LLM, vLLM, and Hugging Face TGI.",
    "Research the practical limits of horizontally scaling relational databases vs NewSQL alternatives like CockroachDB and TiDB."

    ]

    OUTPUT_DIR = "good_examples"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "vllm_good_examples_raw_2000_search.jsonl")

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

    print(f"Loading model with vLLM: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    for i, task in enumerate(TASKS):
        rec_idx = task_to_index[task]
        record = existing_data[rec_idx]
        if record.get("good_example") is not None:
            continue

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        good_example = outputs[0].outputs[0].text.strip()
        if args.ensure_goodbye and not good_example.rstrip().endswith("GOODBYE"):
            good_example = good_example.rstrip() + "\nGOODBYE"
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
