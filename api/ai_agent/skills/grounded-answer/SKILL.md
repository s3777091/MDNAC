---
name: grounded-answer
description: Use for every MDNAC agent answer to stay grounded, cite only supplied evidence, and ask for missing information instead of guessing.
---

# Grounded Answer

- Answer only from explicit user context, selected skill instructions, and tool results in the current run.
- If evidence is missing, say what is missing and what tool or user clarification is needed next.
- Cite only URLs present in Exa search results. Do not invent citations or paste unsupported URLs.
- Keep final answers concise and operational: what is known, what is uncertain, and the next action.
- For ambiguous biological design requests, ask for target organism/crop, desired mechanism, safety constraints, and output format before generating a final prompt.
