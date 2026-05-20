---
applyTo: "reviews/**"
description: "Conventions and severity definitions for all HAMTools review documents. Applied to any file written under the reviews/ directory."
---

# HAMTools Review Conventions

These rules apply to every file written by the Math Reviewer, Code Reviewer, Science Auditor, Doc Reviewer, and Review Orchestrator agents.

## Directory Layout
```
reviews/
  math/       ← Math Reviewer outputs
  code/       ← Code Reviewer outputs
  science/    ← Science Auditor outputs
  docs/       ← Doc Reviewer outputs
  REPORT.md   ← Master consolidated report (Orchestrator only)
```

## Severity Definitions

### Math & Science Severity
| Level | Meaning |
|-------|---------|
| **CRITICAL** | Formula is wrong (wrong sign, wrong index, wrong equation) — produces incorrect output |
| **WARNING** | Ambiguous convention, missing edge case, or result that is correct for common inputs but wrong in degenerate cases |
| **NOTE** | Notation inconsistency, style preference, or cosmetic issue |
| **STRONG** | Exemplary practice that should be preserved or replicated |

### Code Severity
| Level | Meaning |
|-------|---------|
| **BUG** | Will produce wrong numerical output or crash |
| **RISK** | May fail under edge cases (near-zero inputs, large batches, non-square Hessians) |
| **STYLE** | Violates ARCH_SPEC conventions but does not affect correctness |

### Documentation Severity
| Level | Meaning |
|-------|---------|
| **MISSING** | Public API is entirely undocumented |
| **INACCURATE** | Documentation contradicts implementation or spec |
| **UNCLEAR** | Ambiguous to one of the two target audiences (mathematicians / ML engineers) |
| **TYPO** | Minor spelling or formatting error |

## Linking Convention
- Always link to the exact spec section: `spec/MATH_SPEC.md § 2.1` or `spec/ARCH_SPEC.md § 3.2`.
- Always include the source file and line number for every finding: `src/ham/geometry/transport.py:42`.
- For literature references, prefer ArXiv IDs (`arXiv:XXXX.XXXXX`) or DOIs over bare URLs.

## Writing Standards
- Findings must be actionable: every CRITICAL, BUG, FLAW, and MISSING must include a "Recommended Action."
- Do not speculate about intent. State what the code does, then state what the spec or literature says it should do.
- Use LaTeX notation for all mathematics (wrap in `$...$`).
