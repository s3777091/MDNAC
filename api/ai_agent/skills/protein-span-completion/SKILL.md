---
name: protein-span-completion
description: Use for requests that need a protein span-completion instruction/input, protein candidate search, NCBI/ENA retrieval, semantic protein ranking, mask-span selection, or vague biological goals such as crop-yield improvement.
---

# Protein Span Completion

- Clarify vague goals before searching proteins. For "increase crop yield", ask/propose a sharper query around plant-growth mechanisms such as nitrogen fixation, phosphate solubilization, auxin/IAA biosynthesis, ACC deaminase, stress tolerance, or biocontrol.
- Use Exa only for public background research. Use NCBI/ENA retrieval plus local semantic protein ranking for final protein candidate selection.
- Semantic search is the final selection step before producing `instruction` and `input`.
- Rank candidates by biological metadata match, sequence quality, and relevance to the refined query.
- Select a masked span that is inside the protein sequence, uses standard amino acids, has useful flanks, and does not reveal the missing span in the prompt.
- Return only `instruction` and `input` for span-completion prompts; never expose the hidden output span.
