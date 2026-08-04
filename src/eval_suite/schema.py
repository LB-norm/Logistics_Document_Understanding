"""Small dependency-free validator for the JSON Schema features used here."""

from __future__ import annotations

from typing import Any


def _resolve_pointer(root_schema: Any, pointer: str) -> Any:
    if not pointer.startswith("#/"):
        raise ValueError(f"Only local JSON Schema references are supported: {pointer!r}")
    node = root_schema
    for raw_part in pointer[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def _matches_type(value: Any, declared_type: str) -> bool:
    if declared_type == "null":
        return value is None
    if declared_type == "object":
        return isinstance(value, dict)
    if declared_type == "array":
        return isinstance(value, list)
    if declared_type == "string":
        return isinstance(value, str)
    if declared_type == "boolean":
        return isinstance(value, bool)
    if declared_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_json_schema(instance: Any, schema: Any) -> list[str]:
    """Return validation errors for the schema subset used by this project.

    Supported keywords are ``$ref``, ``type``, ``required``, ``properties``,
    ``additionalProperties``, ``items``, ``enum``, ``const``, and the schema
    combinators ``allOf``, ``anyOf``, and ``oneOf``.
    """
    errors: list[str] = []

    def visit(value: Any, node: Any, path: str) -> None:
        if node is True:
            return
        if node is False:
            errors.append(f"{path}: value is forbidden by schema")
            return
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            try:
                resolved = _resolve_pointer(schema, node["$ref"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{path}: invalid schema reference ({exc})")
                return
            visit(value, resolved, path)
            return

        for child_schema in node.get("allOf", []):
            visit(value, child_schema, path)
        for keyword, required_matches in (("anyOf", 1), ("oneOf", 1)):
            alternatives = node.get(keyword)
            if alternatives:
                match_count = sum(
                    not validate_against(value, alternative)
                    for alternative in alternatives
                )
                if (keyword == "anyOf" and match_count < required_matches) or (
                    keyword == "oneOf" and match_count != required_matches
                ):
                    errors.append(f"{path}: does not satisfy {keyword}")

        declared = node.get("type")
        declared_types = [declared] if isinstance(declared, str) else declared
        if isinstance(declared_types, list) and not any(
            _matches_type(value, item)
            for item in declared_types
            if isinstance(item, str)
        ):
            errors.append(
                f"{path}: expected type {declared_types}, got {type(value).__name__}"
            )
            return

        if "const" in node and value != node["const"]:
            errors.append(f"{path}: value does not match const")
        if "enum" in node and value not in node["enum"]:
            errors.append(f"{path}: value is not in enum")

        if isinstance(value, dict):
            required = node.get("required", [])
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: required property is missing")
            properties = node.get("properties", {})
            if isinstance(properties, dict):
                for key, child_value in value.items():
                    if key in properties:
                        visit(child_value, properties[key], f"{path}.{key}")
                    elif node.get("additionalProperties") is False:
                        errors.append(f"{path}.{key}: additional property is not allowed")
                    elif isinstance(node.get("additionalProperties"), dict):
                        visit(child_value, node["additionalProperties"], f"{path}.{key}")

        if isinstance(value, list) and "items" in node:
            for index, child_value in enumerate(value):
                visit(child_value, node["items"], f"{path}[{index}]")

    def validate_against(value: Any, candidate: Any) -> list[str]:
        start = len(errors)
        visit(value, candidate, "$")
        candidate_errors = errors[start:]
        del errors[start:]
        return candidate_errors

    visit(instance, schema, "$")
    return errors
