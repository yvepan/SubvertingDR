# Web Poison Filtering Mechanism

This document explains how `web poison` works in this project: how poisoned
documents enter the candidate pool, how they compete against real web sources,
what is filtered during the planning stage and the research stage, and why the
research stage effectively contains two rounds of filtering.

## 1. Core Takeaway

In this project, web poisoning does not mean publishing documents to the public
internet. Instead, local poisoned documents are converted into virtual
`http://research/...` URLs and mixed into the web retrieval candidate pool.

The mechanism can be summarized as:

```text
local poisoned documents
  -> convert to virtual http://research/{slug}{ext} URLs
  -> mix them into the real web URL candidate pool
  -> run URL-level top-k filtering jointly with real URLs
  -> if selected, the scraper maps http://research/... back to a local file
  -> the research stage then runs chunk-level embedding filtering
  -> only relevant chunks enter the final context / learning / report
```

The most important distinction is:

```text
planning stage: URL-level filtering only
research stage: URL-level filtering first, then content chunk-level filtering
```

## 2. Web Poison Entry Point

CLI argument:

```python
--doc_path_web_poison
```

After it is passed in, the code sets:

```python
os.environ["DOC_PATH_WEB_POISON"] = args.doc_path_web_poison
```

Default configuration:

```python
"DOC_PATH_LOCAL_POISON": "./local-poison-docs",
"DOC_PATH_WEB_POISON": "./web-poison-docs",
```

Typical experimental usage:

```powershell
--doc_path_web_poison <temporary_poison_document_directory>
```

This means the files under that directory will be treated as web-poison
candidate sources.

## 3. How Poisoned Documents Become Virtual Web URLs

Relevant code lives in `gpt_researcher/actions/planning_sources.py`.

Core helper:

```python
def _to_virtual_url(base_name: str, ext: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", base_name).strip("-").lower()
    return f"http://research/{slug}{ext}"
```

For example:

```text
The Negentropic Grid Why Renewable Transition is a Thermodynamic Imperative.md
```

becomes:

```text
http://research/the-negentropic-grid-why-renewable-transition-is-a-thermodynamic-imperative.md
```

Each poisoned document is converted into a candidate item like:

```python
{
    "href": "http://research/xxx.md",
    "title": "filename without extension",
    "body": "first 500 characters of the file",
}
```

Note that `body` is only a short preview used during URL filtering. The full
document is read later during scraping.

Supported formats:

```text
.pdf .txt .doc .docx .pptx .csv .xls .xlsx .md .html
```

## 4. Why `hybrid` Is Required Instead of `web`

Virtual web-poison URLs are mixed in only under this condition:

```python
if researcher.report_source in ["local", "hybrid"]:
```

So:

```text
--report_source web
```

does not mix documents from `DOC_PATH_WEB_POISON` into the candidate pool.

To actually activate web poisoning, use:

```powershell
--report_source hybrid
```

Because `hybrid` also triggers local-document loading logic, if your goal is to
measure only web poison, you should usually pass an explicit empty local-poison
directory as well:

```powershell
--doc_path_local_poison outputs\empty_local_poison_docs
```

## 5. High-Level View of the Filtering Pipeline

There are three filtering layers relevant to web poison in this project:

| Stage | Filter Level | Input | Output | Role |
|---|---|---|---|---|
| Planning stage | URL-level | Real web URLs + `http://research/...` poison URLs | Filtered URL items | Feeds the planning prompt and affects sub-question generation |
| Research stage, first pass | URL-level | Real web URLs + `http://research/...` poison URLs | URLs to scrape | Decides which sources are read |
| Research stage, second pass | Content chunk-level | Full scraped content from web pages and poisoned documents | Relevant context chunks | Decides what enters learning and the final report |

In one sentence:

```text
the planning stage filters source summaries;
the research stage first filters source URLs, then filters source content chunks
```

## 6. Planning Stage: URL-Level Filtering

### 6.1 How the Candidate Pool Is Built

The planning stage first calls the real retriever:

```python
search_results = await asyncio.to_thread(
    retriever.search,
    max_results=researcher.cfg.max_search_results_per_query,
)
```

Real web results are normalized into items such as:

```python
{
    "href": href,
    "title": result.get("title", ""),
    "body": result.get("body", "") or result.get("content", ""),
}
```

Then, if the mode is `local` or `hybrid`, poisoned local documents are added:

```python
if researcher.report_source in ["local", "hybrid"]:
    local_url_items = await get_local_document_url_items(researcher)
    all_url_items.extend(local_url_items)
```

So the final pool is:

```text
all_url_items = real web URL items + web poison URL items
```

### 6.2 Which Query Is Used for Planning-Stage Filtering

The planning stage filters directly using the current planning query:

```python
filtered_items = await filter_url_items_by_title_similarity(
    query,
    all_url_items,
    researcher,
    dynamic_top_k=True,
    summary_max_chars=200,
)
```

At the top level of deep research, this is usually the user query. At recursive
levels, it may be a follow-up query generated by the previous round.

### 6.3 Where the Planning-Stage Result Goes

The filtered planning sources are inserted into the planning prompt and affect
the generation of subsequent research questions.

Important implication:

```text
if a poison URL is selected during planning,
it only means its title/body/url influenced question planning;
it does not mean the full poisoned document was read,
and it does not mean the final report used it
```

## 7. Research Stage, First Pass: URL-Level Filtering

### 7.1 How the Candidate Pool Is Built

For each research subtask, the candidate pool is built again.

It again contains:

```text
all_url_items = real web URL items + web poison URL items
```

### 7.2 Which Query Is Used During Research-Stage URL Filtering

The research stage does not always filter using just the current sub-question.
It typically builds a combined query from:

```text
original user query + current sub-research query
```

This is stricter than the planning stage because it considers both the overall
task and the current subtask.

### 7.3 Deduplication and Shuffling After URL Filtering

After URL-level filtering, the code may:

- remove already visited URLs
- shuffle the order of the remaining URLs before scraping

So even if a poison URL is selected again for a later subtask, it may be
removed if it has already been visited.

## 8. How URL-Level Top-k Selection Actually Works

Both the planning stage and the first research-stage pass use the same URL-level
ranking/filtering logic.

### 8.1 How `k` Is Chosen

The code first counts real URLs and poison URLs separately. The rule is:

```text
k = number of real URLs
```

not:

```text
k = real URLs + poison URLs
```

and not:

```text
k = number of poisoned documents
```

Examples:

```text
real URLs = 5
poison URLs = 3
total candidates = 8
final keep = 5
```

```text
real URLs = 2
poison URLs = 3
total candidates = 5
final keep = 2
```

If the number of real URLs is zero, then all candidates can be returned, and
poison URLs do not need to compete against real URLs.

### 8.2 How Each Candidate Is Scored

All candidates are ranked together:

```text
real URLs + poison URLs
```

The release uses a linear blend of BM25 and embedding similarity:

```text
final_score = bm25_weight * normalized_bm25
            + embedding_weight * normalized_embedding_similarity
```

Default weights:

```text
bm25_weight = 0.4
embedding_weight = 0.6
```

Candidates are sorted by `final_score` and the top `k` items are kept.

### 8.3 What Text Is Ranked

The ranking text is effectively:

```text
candidate title + first 200 characters of candidate body
```

For poisoned documents:

- `title` is the filename without extension
- `body` is derived from the first 500 characters, then truncated to 200 during ranking

So whether a poisoned document enters URL top-k depends heavily on:

- whether its filename matches the query well
- whether the first 200 characters of the document match the query well

### 8.4 BM25 Score

Intuition:

- more query-token overlap in the candidate title/body leads to a higher BM25 score
- rarer matched terms contribute more through IDF
- document length is normalized to avoid favoring long text by default

BM25 scores are normalized before fusion.

### 8.5 Embedding Score

The code computes embedding-based similarity between the query and each
candidate text using cosine similarity, then maps it into `[0, 1]`.

Intuition:

```text
even if keywords do not match exactly,
semantically similar candidates can still receive high scores
```

If embedding computation fails, the code falls back to pure BM25.

### 8.6 Example of URL Top-k Competition

Suppose:

```text
real URLs: A, B, C, D, E
poison URLs: P1, P2, P3
```

After scoring and sorting, perhaps only some poison URLs survive in the top `k`.

This means poison URLs are not appended after the real search results. They
compete in the same pool against real URLs.

## 9. Research Stage, Second Pass: Content Chunk-Level Filtering

The first research-stage pass only decides which URLs to read. After scraping,
the system still needs to decide which content chunks actually enter the context.

The scraped page or poisoned document is split into chunks, and then only chunks
that are sufficiently similar to the current query are kept.

### 9.1 How Content Is Split

The code slices the full text into chunks of roughly:

```text
1000 characters per chunk
100 characters overlap
```

### 9.2 How Chunks Are Filtered

Each chunk must pass an embedding-similarity threshold against the current
sub-question to enter the context candidate set.

### 9.3 How Many Chunks Are Kept

At most a limited number of top chunks are kept for the downstream context.

This second filtering pass is what determines whether poisoned content actually
reaches learning generation and final reporting.

## 10. How the Scraper Reads Full Poisoned Documents

If a URL like `http://research/...` is selected in research-stage URL filtering,
the scraper intercepts it:

```python
if link.startswith("http://research/"):
    return await self._extract_local_document(link)
```

`_extract_local_document()` will:

- read `DOC_PATH_WEB_POISON`
- extract `{slug}{ext}` from the virtual URL
- iterate through the local poison-document directory
- slugify each local filename in the same way
- locate the matching file
- load the full content using the appropriate loader

It returns an object like:

```python
{
    "url": "http://research/xxx.md",
    "raw_content": "full text of the local poisoned document",
    "image_urls": [],
    "title": "real_filename.md",
}
```

## 11. How to Interpret Graph Statistics

`tools/checkgraph.py` mainly uses `//research/` to determine whether web-poison
sources were used.

Typical graph statistics should be interpreted like this:

```text
planning sources used poisoned documents
```

means:

```text
the poison URL passed planning-stage URL filtering
and entered the planning prompt
```

but does not mean:

```text
the full poisoned document was scraped
or its content entered the final report
```

```text
research sources used poisoned documents
```

means:

```text
the poison URL passed the first research-stage filter
and was selected as a source to scrape or record
```

but still does not necessarily mean:

```text
poisoned chunks definitely entered the learning stage
```

```text
learning contains poisoned-source links
```

is stronger evidence:

```text
the poisoned source was not only retrieved,
but its content is more likely to have influenced the LLM's key findings
```

## 12. Why You Can See “Planning Used Poison, Research Did Not”

This can happen because the planning stage and research stage are separate
pipelines.

Planning stage:

```text
filter URLs using the current planning query
then feed results into the planning prompt
```

Research stage:

```text
for each sub-research query, filter URLs again
use a combined query from the original query and the current sub-query
then scrape and run chunk filtering
```

So it is possible that:

```text
a poison URL scores highly during planning,
but scores too low for a later research subtask and is displaced by real URLs
```

It is also possible that:

```text
a poison URL survives the first research-stage pass,
but its full-text chunks are not similar enough to the sub-question,
so nothing enters the final context
```

## 13. How the Current Experiment Script Relates to This Mechanism

Relevant script:

- `tools/run_depth_web_poison_experiment.py`

Key settings:

```python
POISON_DOC_COUNT = 3
DEPTH_VALUES = range(1, 5)
BREADTH_VALUE = 2
BASE_ARGS = [
    "--report_type", "deep",
    "--report_source", "hybrid",
    "--deep_research_breadth", str(BREADTH_VALUE),
    "--no-pdf",
    "--no-docx",
]
```

Poison document selection:

```python
docs.sort(key=lambda item: item.name)
return docs[:POISON_DOC_COUNT]
```

Meaning:

```text
for each query directory,
the script sorts documents by filename
and takes the first three as web-poison candidates
```

Important:

```text
the first three documents only enter the web-poison candidate pool;
they are not guaranteed to survive URL top-k;
they are not guaranteed to appear in the final report
```

They still need to pass:

```text
planning-stage URL filtering
research-stage URL filtering
research-stage chunk filtering
```

## 14. Recommended Experimental Command

If your goal is to test web poison, use:

```powershell
python cli.py "your query" `
  --report_type deep `
  --report_source hybrid `
  --deep_research_depth 1 `
  --deep_research_breadth 2 `
  --doc_path_web_poison "outputs\temp_web_poison_docs\some_query" `
  --doc_path_local_poison "outputs\empty_local_poison_docs" `
  --no-pdf `
  --no-docx
```

Here:

```text
--report_source hybrid
```

is required so the web-poison URLs are mixed into the candidate pool.

And:

```text
--doc_path_local_poison outputs\empty_local_poison_docs
```

is used to avoid introducing extra local-document effects from the hybrid mode.

## 15. Final Summary

The core logic of the web-poison filtering mechanism is:

```text
1. Poisoned documents are first converted into virtual http://research/... URLs.
2. Virtual poison URLs and real web URLs enter the same candidate pool.
3. URL-level filtering uses k = number of real URLs.
4. Real URLs and poison URLs compete jointly by final_score for the top k slots.
5. final_score = 0.4 * BM25 + 0.6 * embedding similarity.
6. Planning-stage results enter the planning prompt.
7. Research-stage selected URLs enter the scraper.
8. The scraper reads the full poisoned document content.
9. The research stage then runs embedding-based chunk filtering over the full text.
10. Only chunks that survive chunk filtering actually enter context and later report generation.
```

So, to judge whether a web-poison document truly influenced the final report,
you need to inspect the whole chain:

```text
Did it enter the candidate pool?
Did it survive planning-stage URL top-k?
Did it survive research-stage URL top-k?
Was it successfully read by the scraper?
Did it survive content chunk filtering?
Did the LLM write it into learning/report?
```
