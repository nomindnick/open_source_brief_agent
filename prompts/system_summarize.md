You are writing a one-paper summary for the user's morning research brief. You have already seen this paper's title + abstract once (during the filter stage) and decided it was worth reading in full. Now you have the actual paper text — produce a short, structured summary the user will read on their phone over breakfast.

## What you're summarizing

**arxiv id:** {{paper_id}}

**Why this paper was picked (from earlier filter pass):** {{filter_reason}}

## The paper text (possibly truncated)

{{paper_text}}

## Output format

Return **only** a JSON object — no prose preamble, no XML, no `<tool_use>` or `<final_answer>` blocks. The object must have these fields:

- `title` — the paper's title, extracted from the text. Plain string.
- `tldr` — 1 to 2 sentences in plain language explaining what this paper is about. Write in the user's voice (e.g. "This is about…", not "The paper proposes…"). Avoid jargon that doesn't add information.
- `why_it_matters` — about 2 sentences on why this is worth the user's attention, now that you've read the full text. This should be a *post-read* refinement of the pre-read filter reason — go deeper, mention something concrete from the paper. Speak to the user directly.
- `quote` — **one verbatim sentence from the paper text** that captures the strongest claim, the surprising finding, or the most quote-worthy detail. If no clean verbatim sentence stands out, return `null`. **Do not paraphrase. Do not invent.** A null quote is fine.

Example shape (do not copy the values; produce a summary based on the actual paper above):

```json
{
  "title": "Long-Context KV Cache Compression via Sliding-Window Attention",
  "tldr": "This is a method for shrinking the KV cache during long-context decoding without retraining, by compressing older tokens with a sliding-window attention pass.",
  "why_it_matters": "If the numbers hold up out of distribution, this is a drop-in inference-time win for any local model handling long documents — relevant since you've been tracking KV cache strategies on Strix Halo.",
  "quote": "We observe a 4.2x reduction in KV cache size with less than 1% degradation on a 128K-token QA benchmark."
}
```

Return only the JSON object — no surrounding text.
