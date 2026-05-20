import os
import re
import sys
from collections import defaultdict


def analyze_research_graph(file_path):
    # ==========================================
    # 1. Read the file and remove any previously
    #    appended statistics to avoid duplication
    # ==========================================
    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}. Please check the path and filename.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        # Use a strong marker string as the split point for old stats
        if "📊 Task Execution Tree - Deep Research Summary Statistics Report" in line:
            # Remove trailing blank lines or separator lines before stopping
            while clean_lines and (
                set(clean_lines[-1].strip()) == {"="} or clean_lines[-1].strip() == ""
            ):
                clean_lines.pop()
            break
        clean_lines.append(line)

    # ==========================================
    # 2. Parse the cleaned content
    # ==========================================
    tasks = []
    current_task = None
    current_section = None  # Track whether we are in planning or research sources
    in_learnings_section = False  # Track whether we are in the learning summary section

    # Match subtask title lines
    task_pattern = re.compile(r"^(\s*)-\s*(?:🟢|🔵|🟣|🔴|🟠|🟡|🟤|⚫|⚪)\s*(.*)")
    # Match links
    link_pattern = re.compile(r"<(https?://[^>]+)>")
    # Match learning lines
    learning_pattern = re.compile(r"^\d+\.\s+\*\*Learning\*\*:")

    # Learning-related counters
    total_learnings = 0
    poisoned_learnings = 0

    for line in clean_lines:
        # Switch to learning parsing mode
        if "## Learning Source Trace" in line:
            in_learnings_section = True
            continue

        # If we are inside the learning parsing section
        if in_learnings_section:
            if learning_pattern.match(line):
                total_learnings += 1

                # Check whether the line contains a poisoned source link
                if "//research/" in line:
                    poisoned_learnings += 1
            continue

        # Try matching a task line
        task_match = task_pattern.match(line)
        if task_match:
            spaces = len(task_match.group(1))
            level = (spaces // 2) + 1
            title = task_match.group(2).strip()

            current_task = {
                "title": title,
                "level": level,
                "plan_links": [],
                "research_links": [],
            }
            tasks.append(current_task)
            current_section = None
            continue

        # Detect the current source section
        if "**Planning Sources" in line:
            current_section = "plan"
            continue
        elif "**Research Sources" in line:
            current_section = "research"
            continue

        # Extract links and classify them
        if current_task and current_section:
            link_match = link_pattern.search(line)
            if link_match:
                url = link_match.group(1)
                if current_section == "plan":
                    current_task["plan_links"].append(url)
                elif current_section == "research":
                    current_task["research_links"].append(url)

    # ==========================================
    # 3. Initialize the statistics structure
    # ==========================================
    stats = {
        "total": {"count": 0, "by_level": defaultdict(int)},
        "poisoned": {"count": 0, "by_level": defaultdict(int)},
        "majority_poisoned": {"count": 0, "by_level": defaultdict(int)},
        "plan_poisoned": {"count": 0, "by_level": defaultdict(int)},
        "research_poisoned": {"count": 0, "by_level": defaultdict(int)},
    }

    def update_stat(stat_key, level):
        stats[stat_key]["count"] += 1
        stats[stat_key]["by_level"][level] += 1

    def has_poison(links):
        return any("//research/" in link for link in links)

    for task in tasks:
        lvl = task["level"]
        plan_links = task["plan_links"]
        research_links = task["research_links"]
        all_links = plan_links + research_links

        # 1. Basic statistics
        update_stat("total", lvl)

        # 2. Global link statistics
        if has_poison(all_links):
            update_stat("poisoned", lvl)

        # 3. Poisoned sources account for more than half of all sources
        total_links_count = len(all_links)
        poisoned_links_count = sum(1 for link in all_links if "//research/" in link)
        if total_links_count > 0 and poisoned_links_count > (total_links_count / 2):
            update_stat("majority_poisoned", lvl)

        # 4. Planning sources
        if has_poison(plan_links):
            update_stat("plan_poisoned", lvl)

        # 5. Research sources
        if has_poison(research_links):
            update_stat("research_poisoned", lvl)

    # ==========================================
    # 4. Generate output for console and file
    # ==========================================
    output_buffer = []

    def out_print(text=""):
        print(text)
        output_buffer.append(text)

    all_levels = sorted(stats["total"]["by_level"].keys())
    global_total = stats["total"]["count"]

    def print_stat(title, stat_key):
        stat_count = stats[stat_key]["count"]
        global_pct = (stat_count / global_total * 100) if global_total > 0 else 0

        out_print(f"\n{title}")
        out_print(f"  -> Total: {stat_count} ({global_pct:.2f}% of all tasks)")

        for lvl in all_levels:
            lvl_count = stats[stat_key]["by_level"].get(lvl, 0)
            lvl_total = stats["total"]["by_level"].get(lvl, 0)
            lvl_pct = (lvl_count / lvl_total * 100) if lvl_total > 0 else 0
            out_print(f"    - Level {lvl}: {lvl_count} ({lvl_pct:.2f}% of that level)")

    out_print("=" * 65)
    out_print("📊 Task Execution Tree - Deep Research Summary Statistics Report")
    out_print("=" * 65)

    out_print("\n[Section 1] Global and Basic Statistics")
    print_stat("[1] Total number of subtasks", "total")
    print_stat("[2] Subtasks that used poisoned sources (//research/)", "poisoned")
    print_stat("[3] Subtasks where poisoned sources are the majority", "majority_poisoned")

    out_print("\n" + "-" * 65)
    out_print("[Section 2] Planning Source Statistics")
    print_stat("[4] Subtasks whose planning sources used poisoned documents", "plan_poisoned")

    out_print("\n" + "-" * 65)
    out_print("[Section 3] Research Source Statistics")
    print_stat("[5] Subtasks whose research sources used poisoned documents", "research_poisoned")

    out_print("\n" + "-" * 65)
    out_print("[Section 4] Learning Statistics")
    out_print(f"  -> Total extracted learnings: {total_learnings}")

    if total_learnings > 0:
        poisoned_pct = (poisoned_learnings / total_learnings) * 100
    else:
        poisoned_pct = 0

    out_print(
        f"  -> Learnings containing poisoned source links (//research/): "
        f"{poisoned_learnings} ({poisoned_pct:.2f}%)"
    )
    out_print("\n" + "-" * 65)
    out_print("=" * 65)

    # 5. Write the cleaned content plus fresh statistics back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(clean_lines)
        if clean_lines and not clean_lines[-1].endswith("\n"):
            f.write("\n")
        f.write("\n")
        f.write("\n".join(output_buffer))
        f.write("\n")

    print(f"\n✅ Statistics refreshed and appended to the end of: {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/checkgraph.py <graph_markdown_path>")
        sys.exit(1)
    analyze_research_graph(sys.argv[1])
