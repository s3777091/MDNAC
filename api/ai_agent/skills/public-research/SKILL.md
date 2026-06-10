---
name: public-research
description: Use when the agent needs current/public background evidence, external citations, or literature context. Use Exa only for public research, not for final protein candidate semantic ranking.
---

# Public Research

- Use `exa_search` when the request asks for latest/current information, citations, sources, literature, or when there is no reliable context.
- Treat Exa results as public background evidence. Summarize them; do not treat them as the final protein database search.
- Prefer search queries that include the biological mechanism, organism/crop, and protein/function names when known.
- Report search failure clearly and continue only from provided context.
- Never cite a URL unless it appears in `search_results`.
