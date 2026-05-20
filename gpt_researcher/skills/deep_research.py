from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta

from defense.root_query_anchor import build_root_query_anchored_query
from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..actions.planning_sources import get_planning_search_results
from ..utils.llm import create_chat_completion
from ..utils.enum import ReportType, ReportSource, Tone
from ..actions.utils import stream_output

logger = logging.getLogger(__name__)

# Maximum words allowed in context (25k words for safety margin)
MAX_CONTEXT_WORDS = 25000

def count_words(text: str) -> int:
    """Count words in a text string"""
    return len(text.split())

def trim_context_to_word_limit(context_list: List[str], max_words: int = MAX_CONTEXT_WORDS) -> List[str]:
    """Trim context list to stay within word limit while preserving most recent/relevant items"""
    total_words = 0
    trimmed_context = []

    # Process in reverse to keep most recent items
    for item in reversed(context_list):
        words = count_words(item)
        if total_words + words <= max_words:
            trimmed_context.insert(0, item)  # Insert at start to maintain original order
            total_words += words
        else:
            break

    return trimmed_context

import math

class ResearchProgress:
    def __init__(self, total_depth: int, total_breadth: int):
        self.current_depth = 1  # Start from 1 and increment up to total_depth
        self.total_depth = total_depth
        self.current_breadth = 0  # Start from 0 and count up to total_breadth as queries complete
        self.total_breadth = total_breadth
        self.current_query: Optional[str] = None
        self.total_queries = 0
        self.completed_queries = 0


class DeepResearchSkill:
    def __init__(self, researcher):
        self.researcher = researcher
        self.breadth = getattr(researcher.cfg, 'deep_research_breadth', 4)
        self.depth = getattr(researcher.cfg, 'deep_research_depth', 2)
        self.concurrency_limit = getattr(researcher.cfg, 'deep_research_concurrency', 2)
        self.max_context_words = int(getattr(researcher.cfg, 'max_context_words', MAX_CONTEXT_WORDS))
        self.enable_defense = bool(getattr(researcher.cfg, 'enable_deep_research_defense', False))
        self.websocket = researcher.websocket
        self.tone = researcher.tone
        self.config_path = researcher.cfg.config_path if hasattr(researcher.cfg, 'config_path') else None
        self.headers = researcher.headers or {}
        self.visited_urls = researcher.visited_urls
        self.learnings = []
        self.research_sources = []  # Track all research sources
        self.context = []  # Track all context

    def _anchor_query(self, root_query: str | None, query: str) -> str:
        if not self.enable_defense or not root_query:
            return query
        anchored_query = build_root_query_anchored_query(root_query, query)
        if anchored_query != query:
            logger.info(f"Deep research defense: root-query anchoring enabled root={root_query[:80]!r}")
        return anchored_query

    async def generate_search_queries(
        self,
        query: str,
        num_queries: int = 3,
        target_task: str = None,
        root_query: str | None = None,
    ) -> tuple[List[Dict[str, str]], dict]:
        """Generate SERP queries for research

        Args:
            query: The query to generate search queries for
            num_queries: Number of queries to generate
            target_task: Optional task name to record planning costs to

        Returns:
            Tuple of (queries list, cost_info dict with tokens and elapsed_ms)
        """
        # Track costs for this specific call
        planning_tokens = 0
        planning_elapsed_ms = 0
        planning_search_results: list[dict[str, Any]] = []

        def cost_tracker(cost):
            nonlocal planning_tokens, planning_elapsed_ms
            if isinstance(cost, dict):
                planning_tokens += cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0)
                planning_elapsed_ms += cost.get("elapsed_ms", 0)
            # Don't record costs here - let the caller decide where to record

        planning_query = self._anchor_query(root_query, query)

        if self.researcher.report_source != ReportSource.Local.value:
            try:
                planning_search_results = await get_planning_search_results(
                    planning_query,
                    self.researcher,
                )
                logger.info(f"Recursive planning stage: obtained {len(planning_search_results)} candidate sources")

                if self.researcher.verbose and planning_search_results:
                    for result in planning_search_results:
                        href = None
                        if isinstance(result, dict):
                            href = result.get("href") or result.get("url")
                        if href:
                            await stream_output(
                                "logs",
                                "planning_source_url",
                                f"🗺️ Planning source: {href}\n",
                                self.researcher.websocket,
                                True,
                                {"url": href, "task": target_task or query},
                            )
            except Exception as e:
                logger.warning(f"Recursive planning stage failed to obtain candidate sources: {e}")
                planning_search_results = []

        planning_context = ""
        if planning_search_results:
            planning_context = "\n\nAvailable Planning Sources:\n" + "\n".join(
                [
                    f"- {item.get('title', '')}: {item.get('body', '')[:300]} ({item.get('href') or item.get('url', '')})"
                    for item in planning_search_results
                ]
            )

        messages = [
            {"role": "system", "content": "You are an expert researcher generating search queries."},
            {"role": "user",
             "content": (
                 f"Given the following prompt, generate {num_queries} unique search queries to research the topic thoroughly. "
                 f"For each query, provide a research goal. Format as 'Query: <query>' followed by 'Goal: <goal>' for each pair.\n\n"
                 f"Prompt:\n{planning_query}"
                 f"{planning_context}"
             )}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=self.researcher.cfg.reasoning_effort,
            temperature=0.4,
            cost_callback=cost_tracker
        )

        lines = response.split('\n')
        queries = []
        current_query = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Query:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query': line.replace('Query:', '').strip()}
            elif line.startswith('Goal:') and current_query:
                current_query['researchGoal'] = line.replace('Goal:', '').strip()

        if current_query:
            queries.append(current_query)

        cost_info = {
            "tokens": planning_tokens,
            "elapsed_ms": planning_elapsed_ms
        }

        return queries[:num_queries], cost_info

    async def generate_research_plan(self, query: str, num_questions: int = 3, root_query: str | None = None) -> List[str]:
        """Generate follow-up questions to clarify research direction"""
        from ..document import DocumentLoader, OnlineDocumentLoader
        from ..utils.enum import ReportSource

        planning_query = self._anchor_query(root_query, query)
        
        # Load local documents if report_source is local or hybrid
        local_context = ""
        if self.researcher.report_source in [ReportSource.Local.value, ReportSource.Hybrid.value]:
            try:
                # Prefer the explicit local-poison path
                doc_path_to_use = self.researcher.cfg.doc_path
                if hasattr(self.researcher.cfg, 'doc_path_local_poison') and self.researcher.cfg.doc_path_local_poison:
                    doc_path_to_use = self.researcher.cfg.doc_path_local_poison
                    logger.info(f"Planning stage: using local-poison path {doc_path_to_use}")
                else:
                    logger.info(f"Planning stage: using default path {doc_path_to_use}")

                if self.researcher.document_urls:
                    document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
                else:
                    document_data = await DocumentLoader(doc_path_to_use).load()
                
                used_document_data = document_data[:5] if document_data else []

                if self.researcher.verbose and used_document_data:
                    try:
                        for doc in used_document_data:
                            if isinstance(doc, dict) and doc.get("url"):
                                await stream_output(
                                    "logs",
                                    "planning_source_url",
                                    f"🗺️ Planning source: {doc['url']}\n",
                                    self.researcher.websocket,
                                    True,
                                    {"url": doc["url"], "task": query},
                                )
                    except Exception:
                        pass

                # Extract content from documents
                local_content_parts = [doc.get('raw_content', '') for doc in used_document_data if doc.get('raw_content')]
                if local_content_parts:
                    local_context = "\n\nLocal Documents Context:\n" + "\n---\n".join(local_content_parts[:5])  # Limit to first 5 docs
                    logger.info(f"Loaded {len(document_data)} local documents for planning")
            except Exception as e:
                logger.warning(f"Failed to load local documents for planning: {e}")
                local_context = ""
        
        # Get initial search results to inform query generation (only if not local-only)
        search_results = ""
        if self.researcher.report_source != ReportSource.Local.value:
            search_results = await get_planning_search_results(
                planning_query,
                self.researcher,
            )
            logger.info(f"Initial web knowledge obtained: {len(search_results)} results")

            if self.researcher.verbose and search_results:
                try:
                    for r in search_results:
                        href = None
                        if isinstance(r, dict):
                            href = r.get("href") or r.get("url")
                        if href:
                            await stream_output(
                                "logs",
                                "planning_source_url",
                                f"🗺️ Planning source: {href}\n",
                                self.researcher.websocket,
                                True,
                                {"url": href, "task": query},
                            )
                except Exception:
                    pass

        # Get current time for context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Combine search results and local context
        combined_context = ""
        if search_results:
            combined_context += f"Web Search Results:\n{search_results}\n"
        if local_context:
            combined_context += local_context

        messages = [
            {"role": "system", "content": "You are an expert researcher. Your task is to analyze the original query and available information, then generate targeted questions that explore different aspects and time periods of the topic."},
            {"role": "user",
             "content": f"""Original query: {planning_query}

Current time: {current_time}

Available Information:
{combined_context}

Based on this information, the original query, and the current time, generate {num_questions} unique questions. Each question should explore a different aspect or time period of the topic, considering recent developments up to {current_time}.

Format each question on a new line starting with 'Question: '"""}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            reasoning_effort=ReasoningEfforts.High.value,
            temperature=0.4,
            cost_callback=lambda cost: self.researcher.add_costs(cost, phase="planning")
        )

        questions = [q.replace('Question:', '').strip()
                     for q in response.split('\n')
                     if q.strip().startswith('Question:')]
        return questions[:num_questions]

    async def process_research_results(self, query: str, context: str, num_learnings: int = 3) -> Dict[str, List[str]]:
        """Process research results to extract learnings and follow-up questions"""
        messages = [
            {"role": "system", "content": "You are an expert researcher analyzing search results."},
            {"role": "user",
             "content": f"Given the following research results for the query '{query}', extract key learnings and suggest follow-up questions. For each learning, include a citation to the source URL if available. Format each learning as 'Learning [source_url]: <insight>' and each question as 'Question: <question>':\n\n{context}"}
        ]

        response = await create_chat_completion(
            messages=messages,
            llm_provider=self.researcher.cfg.strategic_llm_provider,
            model=self.researcher.cfg.strategic_llm_model,
            temperature=0.4,
            reasoning_effort=ReasoningEfforts.High.value,
            max_tokens=1000,
            cost_callback=lambda cost: self.researcher.add_costs(cost, phase="research")
        )

        lines = response.split('\n')
        learnings = []
        questions = []
        citations = {}

        def virtualize_url(url: str) -> str:
            if isinstance(url, str) and url.lower().startswith("file://"):
                name = os.path.basename(url.replace("file://", ""))
                base, _ = os.path.splitext(name)
                slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", base).strip("-").lower()
                return f"https://docs.jarvislabs.ai/blog/{slug}"
            return url

        for line in lines:
            line = line.strip()
            if line.startswith('Learning'):
                url_match = re.search(r'\[(.*?)\]:', line)
                if url_match:
                    url = virtualize_url(url_match.group(1))
                    learning = line.split(':', 1)[1].strip()
                    learnings.append(learning)
                    citations[learning] = url
                else:
                    # Try to find URL in the line itself
                    url_match = re.search(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                    if url_match:
                        url = virtualize_url(url_match.group(0))
                        learning = line.replace(url, '').replace('Learning:', '').strip()
                        learnings.append(learning)
                        citations[learning] = url
                    else:
                        learnings.append(line.replace('Learning:', '').strip())
            elif line.startswith('Question:'):
                questions.append(line.replace('Question:', '').strip())

        return {
            'learnings': learnings[:num_learnings],
            'followUpQuestions': questions[:num_learnings],
            'citations': citations
        }

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            learnings: List[str] = None,
            citations: Dict[str, str] = None,
            visited_urls: Set[str] = None,
            on_progress=None,
            parent_query: str = None,
            root_query: str | None = None,
            target_researcher=None,
            learning_sources: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Conduct deep iterative research"""
        print(f"\n📊 DEEP RESEARCH: depth={depth}, breadth={breadth}, query={query[:100]}...", flush=True)
        if learnings is None:
            learnings = []
        if citations is None:
            citations = {}
        if visited_urls is None:
            visited_urls = set()
        if learning_sources is None:
            learning_sources = []
        if root_query is None:
            root_query = self.researcher.query

        progress = ResearchProgress(depth, breadth)

        if on_progress:
            on_progress(progress)

        # Generate search queries and track planning cost
        print(f"🔎 Generating {breadth} search queries...", flush=True)
        planning_task = target_researcher.query if target_researcher else (parent_query if parent_query else self.researcher.query)
        anchored_query = self._anchor_query(root_query, query)
        serp_queries, planning_cost = await self.generate_search_queries(
            anchored_query,
            num_queries=breadth,
            target_task=planning_task,
            root_query=root_query,
        )
        print(f"✅ Generated {len(serp_queries)} queries: {[q['query'] for q in serp_queries]}", flush=True)
        progress.total_queries = len(serp_queries)

        # Record planning cost to the target researcher (or current task if not specified)
        # This allows recording planning tokens at different recursion levels
        planner = target_researcher if target_researcher else self.researcher
        if planning_cost["tokens"] > 0:
            planner.add_costs({
                "prompt_tokens": planning_cost["tokens"] // 2,
                "completion_tokens": planning_cost["tokens"] // 2,
                "elapsed_ms": planning_cost["elapsed_ms"]
            }, phase="planning")

        all_learnings = learnings.copy()
        all_citations = citations.copy()
        all_visited_urls = visited_urls.copy()
        all_learning_sources = learning_sources.copy()
        all_context = []
        all_sources = []

        # Process queries with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def process_query(serp_query: Dict[str, str]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    progress.current_query = serp_query['query']
                    if on_progress:
                        on_progress(progress)

                    from .. import GPTResearcher
                    # Use parent_query if provided, otherwise don't set it (will default to empty string)
                    # This ensures the hierarchy is correctly tracked in logs
                    # Inherit report_source from parent researcher to support local/hybrid modes
                    researcher = GPTResearcher(
                        query=serp_query['query'],
                        report_type=ReportType.ResearchReport.value,
                        report_source=self.researcher.report_source,  # Inherit from parent
                        tone=self.tone,
                        websocket=self.websocket,
                        config_path=self.config_path,
                        headers=self.headers,
                        visited_urls=self.visited_urls,
                        parent_query=parent_query if parent_query else "",
                        verbose=self.researcher.verbose,  # Propagate verbose setting
                        # Propagate MCP configuration to nested researchers
                        mcp_configs=self.researcher.mcp_configs,
                        mcp_strategy=self.researcher.mcp_strategy,
                        # Propagate log handler to nested researchers
                        log_handler=self.researcher.log_handler
                    )

                    # Conduct research (no planning cost allocation - planning is recorded by the parent task)
                    context = await researcher.conduct_research()

                    # Get results and visited URLs
                    visited = researcher.visited_urls
                    sources = researcher.research_sources

                    # Process results to extract learnings and citations
                    results = await self.process_research_results(
                        query=serp_query['query'],
                        context=context
                    )

                    # Update progress
                    progress.completed_queries += 1
                    progress.current_breadth += 1
                    if on_progress:
                        on_progress(progress)

                    return {
                        'query': serp_query['query'],
                        'researcher': researcher,  # Include researcher instance for recursive planning
                        'learnings': results['learnings'],
                        'visited_urls': list(visited),
                        'followUpQuestions': results['followUpQuestions'],
                        'researchGoal': serp_query['researchGoal'],
                        'citations': results['citations'],
                        'context': context if context else "",
                        'sources': sources if sources else []
                    }

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error processing query '{serp_query['query']}': {str(e)}")
                    print(f"\n❌ DEEP RESEARCH ERROR: {str(e)}\n{error_details}", flush=True)
                    return None

        # Process queries concurrently with limit
        tasks = [process_query(query) for query in serp_queries]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Update breadth progress based on successful queries
        progress.current_breadth = len(results)
        if on_progress:
            on_progress(progress)

        # Collect all results
        for result in results:
            # Record the source subtask for each learning
            for learning in result['learnings']:
                all_learning_sources.append({
                    'learning': learning,
                    'subtask': result['query']
                })
            all_learnings.extend(result['learnings'])
            all_visited_urls.update(result['visited_urls'])
            all_citations.update(result['citations'])
            if result['context']:
                all_context.append(result['context'])
            if result['sources']:
                all_sources.extend(result['sources'])

            # Continue deeper if needed
            if depth > 1:
                # Use logarithmic decay for breadth instead of hard division by 2
                fraction = math.log(depth) / math.log(self.depth + 1) if self.depth > 0 else 0.5
                new_breadth = max(2, math.ceil(self.breadth * fraction))
                new_depth = depth - 1
                progress.current_depth += 1

                # Create next query from research goal and follow-up questions
                next_query = f"""
                Previous research goal: {result['researchGoal']}
                Follow-up questions: {' '.join(result['followUpQuestions'])}
                """

                # Recursive research - pass the sub-task's researcher to record its planning tokens
                deeper_results = await self.deep_research(
                    query=next_query,
                    breadth=new_breadth,
                    depth=new_depth,
                    learnings=all_learnings,
                    citations=all_citations,
                    visited_urls=all_visited_urls,
                    learning_sources=all_learning_sources,
                    on_progress=on_progress,
                    parent_query=result['query'],  # Pass the current query as parent for next depth
                    root_query=root_query,
                    target_researcher=result['researcher']  # Record planning tokens to this sub-task
                )

                all_learnings = deeper_results['learnings']
                all_visited_urls.update(deeper_results['visited_urls'])
                all_citations.update(deeper_results['citations'])
                all_learning_sources = deeper_results.get('learning_sources', [])
                if deeper_results.get('context'):
                    all_context.extend(deeper_results['context'])
                if deeper_results.get('sources'):
                    all_sources.extend(deeper_results['sources'])

        # Update class tracking
        self.context.extend(all_context)
        self.research_sources.extend(all_sources)

        # Trim context to stay within word limits
        trimmed_context = trim_context_to_word_limit(all_context, max_words=self.max_context_words)
        logger.info(f"Trimmed context from {len(all_context)} items to {len(trimmed_context)} items to stay within word limit")

        return {
            'learnings': list(set(all_learnings)),
            'visited_urls': list(all_visited_urls),
            'citations': all_citations,
            'learning_sources': all_learning_sources,
            'context': trimmed_context,
            'sources': all_sources
        }

    async def run(self, on_progress=None) -> str:
        """Run the deep research process and generate final report"""
        print(f"\n🔍 DEEP RESEARCH: Starting with breadth={self.breadth}, depth={self.depth}, concurrency={self.concurrency_limit}", flush=True)
        start_time = time.time()

        # Log initial costs
        initial_costs = self.researcher.get_costs()

        follow_up_questions = await self.generate_research_plan(self.researcher.query, root_query=self.researcher.query)
        answers = ["Automatically proceeding with research"] * len(follow_up_questions)

        qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)]
        combined_query = f"""
        Initial Query: {self.researcher.query}\nFollow - up Questions and Answers:\n
        """ + "\n".join(qa_pairs)

        results = await self.deep_research(
            query=combined_query,
            breadth=self.breadth,
            depth=self.depth,
            on_progress=on_progress,
            parent_query=self.researcher.query,
            root_query=self.researcher.query,
        )

        # Get costs after deep research
        research_costs = self.researcher.get_costs() - initial_costs

        # Log research costs if we have a log handler
        if self.researcher.log_handler:
            await self.researcher._log_event("research", step="deep_research_costs", details={
                "research_costs": research_costs,
                "total_costs": self.researcher.get_costs()
            })

        # Prepare context with citations
        context_with_citations = []
        for learning in results['learnings']:
            citation = results['citations'].get(learning, '')
            if citation:
                context_with_citations.append(f"{learning} [Source: {citation}]")
            else:
                context_with_citations.append(learning)

        # Add all research context
        if results.get('context'):
            context_with_citations.extend(results['context'])

        # Trim final context to word limit
        final_context = trim_context_to_word_limit(context_with_citations, max_words=self.max_context_words)
        
        # Set enhanced context and visited URLs
        self.researcher.context = "\n".join(final_context)
        self.researcher.visited_urls = results['visited_urls']

        # Set research sources
        if results.get('sources'):
            self.researcher.research_sources = results['sources']

        # Log learning sources for graph generation
        if self.researcher.log_handler and results.get('learning_sources'):
            await self.researcher._log_event("research", step="learning_sources", details={
                "learning_sources": results['learning_sources']
            })

        # Log total execution time
        end_time = time.time()
        execution_time = timedelta(seconds=end_time - start_time)
        logger.info(f"Total research execution time: {execution_time}")
        logger.info(f"Total research costs: ${research_costs:.2f}")

        # Return the context - don't generate report here as it will be done by the main agent
        return self.researcher.context
