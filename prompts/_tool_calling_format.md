# Tool calling format

You communicate with the system by emitting **XML blocks** in your response.
Two block types exist: `<tool_use>` for invoking a tool, and `<final_answer>`
for ending the task.

## Calling a tool

To call a tool, emit a `<tool_use>` block containing a `<name>` and an
`<input>`. The `<input>` body must be a valid JSON object with the
arguments the tool expects.

Example:

```
<tool_use>
  <name>read_paper</name>
  <input>{"id": "2509.05591"}</input>
</tool_use>
```

You may emit **more than one** `<tool_use>` block in a single response.
Each will be executed in order, and you will receive the results back
as observations before your next turn.

You may also include freeform reasoning text *outside* the XML blocks
(e.g. before or between them); the system will ignore that text. It
will only act on the `<tool_use>` and `<final_answer>` blocks themselves.

## Ending the task

When you have enough information to answer, emit a single
`<final_answer>` block:

```
<final_answer>
The three papers most relevant to the user's interests today are: ...
</final_answer>
```

After a `<final_answer>` is parsed, the run ends — no further tool calls
will be executed. Do not emit any more tool calls in the same response;
they will be ignored.

## Rules the parser follows

- **JSON only inside `<input>`.** It must parse as a JSON *object* — not
  a list, not a bare value. The parser is lenient about *formatting*:
  trailing commas, single quotes, and surrounding whitespace are all
  cleaned up before strict parsing. It is **strict about structure**:
  if `<input>` cannot be parsed as a JSON object, you'll get a
  `ParseError` observation with the reason.
- **No XML at all means final answer.** If your response contains
  neither a `<tool_use>` nor a `<final_answer>` block, the system treats
  your whole response as a final answer. Use this only when you're
  truly done — otherwise emit a proper tool call.
- **Tools you call must exist.** If `<name>` doesn't match a registered
  tool, you'll get an error observation listing the available tools.

## Recovering from parse errors

If a tool call you emit fails to parse, the next observation will
describe what went wrong (e.g. "input was not a JSON object"). Read it,
fix the format, and try again on your next turn. Do not give up — most
parse errors are simple formatting issues that resolve on retry.
