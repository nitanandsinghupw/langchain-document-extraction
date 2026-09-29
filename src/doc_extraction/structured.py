"""Get validated Pydantic objects from any chat model.

Models with native structured output (tool calling / JSON schema) use it. Others get the JSON schema in
the prompt, and an invalid reply is sent back once with the validation error so the model can fix it.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model did not produce valid output for the schema, even after a repair attempt."""


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may include code fences or extra words."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object found in the reply")
    return match.group(0)


def structured_call(model: BaseChatModel, schema: type[T], system: str, user: str, max_attempts: int = 2) -> T:
    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]

    # Function calling (non-strict) rather than OpenAI's strict JSON-schema mode: strict mode rejects the
    # regex pattern Pydantic generates for Decimal fields. Pydantic validates the result either way.
    try:
        try:
            runnable = model.with_structured_output(schema, method="function_calling")
        except TypeError:  # models whose with_structured_output has no "method" argument
            runnable = model.with_structured_output(schema)
    except NotImplementedError:
        runnable = None

    if runnable is not None:
        result = runnable.invoke(messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        raise StructuredOutputError(f"{schema.__name__}: model returned no structured output")

    schema_json = json.dumps(schema.model_json_schema())
    messages[0] = SystemMessage(
        f"{system}\n\nReply with only a JSON object that matches this JSON schema:\n{schema_json}"
    )
    error: Exception | None = None
    for _ in range(max_attempts):
        reply = model.invoke(messages)
        text = _text(reply)
        try:
            return schema.model_validate_json(extract_json(text))
        except (ValueError, ValidationError) as exc:
            error = exc
            messages += [
                AIMessage(text),
                HumanMessage(f"That reply was invalid: {exc}. Reply again with corrected JSON only."),
            ]
    raise StructuredOutputError(f"{schema.__name__}: invalid output after {max_attempts} attempts ({_describe(error)})")


def _describe(error: Exception | None) -> str:
    """A one-line summary instead of Pydantic's multi-line report."""
    if isinstance(error, ValidationError):
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in error.errors()})
        return f"{error.error_count()} validation errors: {', '.join(fields[:5])}"
    return str(error)
