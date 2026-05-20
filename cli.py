"""
Provides a command line interface for the GPTResearcher class.

Usage:

```shell
python cli.py "<query>" --report_type <report_type> --tone <tone> --query_domains <foo.com,bar.com>
```

"""
import asyncio
import argparse
from argparse import RawTextHelpFormatter
import os

from dotenv import load_dotenv

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import ReportType, ReportSource, Tone
from backend.report_type import DetailedReport
from backend.utils import write_md_to_pdf, write_md_to_word
from backend.server.server_utils import CustomLogsHandler

# =============================================================================
# CLI
# =============================================================================

cli = argparse.ArgumentParser(
    description="Generate a research report.",
    # Enables the use of newlines in the help message
    formatter_class=RawTextHelpFormatter)

# =====================================
# Arg: Query
# =====================================

cli.add_argument(
    # Position 0 argument
    "query",
    type=str,
    help="The query to conduct research on.")

# =====================================
# Arg: Report Type
# =====================================

choices = [report_type.value for report_type in ReportType]

report_type_descriptions = {
    ReportType.ResearchReport.value: "Summary - Short and fast (~2 min)",
    ReportType.DetailedReport.value: "Detailed - In depth and longer (~5 min)",
    ReportType.ResourceReport.value: "",
    ReportType.OutlineReport.value: "",
    ReportType.CustomReport.value: "",
    ReportType.SubtopicReport.value: "",
    ReportType.DeepResearch.value: "Deep Research"
}

cli.add_argument(
    "--report_type",
    type=str,
    help="The type of report to generate. Options:\n" + "\n".join(
        f"  {choice}: {report_type_descriptions[choice]}" for choice in choices
    ),
    # Deserialize ReportType as a List of strings:
    choices=choices,
    required=True)

# =====================================
# Arg: Tone
# =====================================

cli.add_argument(
    "--tone",
    type=str,
    help="The tone of the report (optional).",
    choices=["objective", "formal", "analytical", "persuasive", "informative",
            "explanatory", "descriptive", "critical", "comparative", "speculative",
            "reflective", "narrative", "humorous", "optimistic", "pessimistic"],
    default="objective"
)

# =====================================
# Arg: Encoding
# =====================================

cli.add_argument(
    "--encoding",
    type=str,
    help="The encoding to use for the output file (default: utf-8).",
    default="utf-8"
)

# =====================================
# Arg: Query Domains
# =====================================

cli.add_argument(
    "--query_domains",
    type=str,
    help="A comma-separated list of domains to search for the query.",
    default=""
)

# =====================================
# Arg: Report Source
# =====================================

cli.add_argument(
    "--report_source",
    type=str,
    help="The source of information for the report.",
    choices=["web", "local", "hybrid", "azure", "langchain_documents",
             "langchain_vectorstore", "static"],
    default="web"
)

# =====================================
# Arg: Output Format Flags
# =====================================

cli.add_argument(
    "--no-pdf",
    action="store_true",
    help="Skip PDF generation (generate markdown and DOCX only)."
)

cli.add_argument(
    "--no-docx",
    action="store_true",
    help="Skip DOCX generation (generate markdown and PDF only)."
)

# =====================================
# Arg: Poison Docs Paths
# =====================================

cli.add_argument(
    "--doc_path_local_poison",
    type=str,
    help="Path to local poison documents directory (default from config).",
    default=None
)

cli.add_argument(
    "--doc_path_web_poison",
    type=str,
    help="Path to web poison documents directory (default from config).",
    default=None
)

defense_group = cli.add_mutually_exclusive_group()
defense_group.add_argument(
    "--enable-defense",
    dest="enable_defense",
    action="store_true",
    help="Enable root-query anchoring in deep research.",
)
defense_group.add_argument(
    "--disable-defense",
    dest="enable_defense",
    action="store_false",
    help="Disable root-query anchoring in deep research.",
)
cli.set_defaults(enable_defense=None)

# =====================================
# Arg: Deep Research Settings
# =====================================

cli.add_argument(
    "--deep_research_depth",
    type=int,
    help="Depth of deep research iterations.",
    default=None
)

cli.add_argument(
    "--deep_research_breadth",
    type=int,
    help="Breadth of concurrent queries per depth level in deep research.",
    default=None
)

cli.add_argument(
    "--deep_research_concurrency",
    type=int,
    help="Concurrency limit for deep research.",
    default=None
)

cli.add_argument(
    "--max_context_words",
    type=int,
    help="Maximum context words allowed for deep research.",
    default=None
)

# =====================================
# Arg: Config File
# =====================================

cli.add_argument(
    "--config_file",
    type=str,
    help="Path to a custom JSON configuration file.",
    default=None
)

# =============================================================================
# Main
# =============================================================================

async def main(args):
    """
    Conduct research on the given query, generate the report, and write
    it as a markdown file to the output directory.
    """
    # Override poison document paths via environment variables if provided
    if args.doc_path_local_poison:
        os.environ["DOC_PATH_LOCAL_POISON"] = args.doc_path_local_poison
    if args.doc_path_web_poison:
        os.environ["DOC_PATH_WEB_POISON"] = args.doc_path_web_poison
    if args.enable_defense is not None:
        os.environ["ENABLE_DEEP_RESEARCH_DEFENSE"] = "true" if args.enable_defense else "false"

    # Override deep research specific configurations via environment variables
    if args.deep_research_depth is not None:
        os.environ["DEEP_RESEARCH_DEPTH"] = str(args.deep_research_depth)
    if args.deep_research_breadth is not None:
        os.environ["DEEP_RESEARCH_BREADTH"] = str(args.deep_research_breadth)
    if args.deep_research_concurrency is not None:
        os.environ["DEEP_RESEARCH_CONCURRENCY"] = str(args.deep_research_concurrency)
    if args.max_context_words is not None:
        os.environ["MAX_CONTEXT_WORDS"] = str(args.max_context_words)

    query_domains = args.query_domains.split(",") if args.query_domains else []

    logs_handler = CustomLogsHandler(None, args.query)
    await logs_handler.send_json({
        "query": args.query,
        "sources": [],
        "context": [],
        "report": ""
    })

    if args.report_type == 'detailed_report':
        detailed_report = DetailedReport(
            query=args.query,
            query_domains=query_domains,
            report_type="research_report",
            report_source="web_search",
            websocket=logs_handler,
        )

        report = await detailed_report.run()
    else:
        # Convert the simple keyword to the full Tone enum value
        tone_map = {
            "objective": Tone.Objective,
            "formal": Tone.Formal,
            "analytical": Tone.Analytical,
            "persuasive": Tone.Persuasive,
            "informative": Tone.Informative,
            "explanatory": Tone.Explanatory,
            "descriptive": Tone.Descriptive,
            "critical": Tone.Critical,
            "comparative": Tone.Comparative,
            "speculative": Tone.Speculative,
            "reflective": Tone.Reflective,
            "narrative": Tone.Narrative,
            "humorous": Tone.Humorous,
            "optimistic": Tone.Optimistic,
            "pessimistic": Tone.Pessimistic
        }

        researcher = GPTResearcher(
            query=args.query,
            query_domains=query_domains,
            report_type=args.report_type,
            report_source=args.report_source,
            tone=tone_map[args.tone],
            encoding=args.encoding,
            websocket=logs_handler,
            log_handler=logs_handler,
            config_path=args.config_file
        )

        await researcher.conduct_research()

        report = await researcher.write_report()

    await logs_handler.send_json({
        "report": report
    })

    artifact_base = os.path.splitext(os.path.basename(logs_handler.log_file))[0]
    artifact_filepath = os.path.join("outputs", f"{artifact_base}.md")
    os.makedirs("outputs", exist_ok=True)
    with open(artifact_filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to '{artifact_filepath}'")

    # Generate PDF if not disabled
    if not args.no_pdf:
        try:
            pdf_path = await write_md_to_pdf(report, artifact_base)
            if pdf_path:
                print(f"PDF written to '{pdf_path}'")
        except Exception as e:
            print(f"Warning: PDF generation failed: {e}")

    # Generate DOCX if not disabled
    if not args.no_docx:
        try:
            docx_path = await write_md_to_word(report, artifact_base)
            if docx_path:
                print(f"DOCX written to '{docx_path}'")
        except Exception as e:
            print(f"Warning: DOCX generation failed: {e}")

    graph_path = logs_handler.generate_task_graph({"report_type": args.report_type})
    print(f"Graph written to '{graph_path}'")

    # Automatically run analyze_research_graph on the generated graph
    try:
        from tools.checkgraph import analyze_research_graph
        print(f"Analyzing generated graph file: '{graph_path}'")
        analyze_research_graph(graph_path)
    except Exception as e:
        import traceback
        print(f"Warning: Failed to execute graph analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    args = cli.parse_args()
    asyncio.run(main(args))
