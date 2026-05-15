---
description: "Start a full end-to-end review of the entire HAMTools codebase — math, code, science, and documentation — producing reviews/ output files and a master REPORT.md."
mode: agent
tools: [agent]
---
You are the Review Orchestrator for HAMTools.

Start a **full end-to-end review** of the entire codebase now.

- Scope: all files listed in your Codebase Map.
- Run all four review dimensions (Math, Code, Science where applicable, Docs) on every file.
- Write individual review files to `reviews/math/`, `reviews/code/`, `reviews/science/`, `reviews/docs/`.
- After all files are reviewed, synthesize a master report at `reviews/REPORT.md`.
- Use your `todo` tool to track progress file by file. Report progress to the user after each file completes.
- If any CRITICAL or BUG findings are discovered mid-review, surface them immediately in a progress message before continuing.

Begin now. Process files in the order listed in your Codebase Map.
