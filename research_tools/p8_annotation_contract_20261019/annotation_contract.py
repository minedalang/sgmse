"""Synthetic-only, conservative source-label translation; no corpus ingestion.

This schema records source assertions, not acoustic truth or enhancement errors.
ASCII letter/digit symbols with an optional terminal '*' are a structural
contract, NOT a validated inventory of any real corpus. Unknown encodings fail.
"""
from decimal import Decimal, InvalidOperation
import re

_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9]*\*?", re.ASCII)
_KINDS = {"manual_annotation", "forced_alignment"}


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise TypeError(f"{name} must be str, int, float or Decimal (not bool)")
    if isinstance(value, str) and (not value or value != value.strip()):
        raise ValueError(f"{name} must be a nonempty unpadded decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} is not decimal-compatible") from None
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def translate_annotation(raw_label, *, source_kind, source_id, start_s, end_s):
    """Translate one explicitly supplied label without I/O or inferred repair.

    `annotation_operation` concerns the SOURCE pronunciation versus its expected
    pronunciation, never an enhancement deletion/insertion. All returned records
    remain ineligible for hard identity supervision and unapproved for real data.
    """
    if not isinstance(raw_label, str):
        raise TypeError("raw_label must be a string")
    if not isinstance(source_kind, str):
        raise TypeError("source_kind must be a string")
    if source_kind not in _KINDS:
        raise ValueError("unsupported source_kind")
    if not isinstance(source_id, str):
        raise TypeError("source_id must be a string")
    if not source_id.strip():
        raise ValueError("source_id must be nonempty")
    start, end = _time(start_s, "start_s"), _time(end_s, "end_s")
    if end < start:
        raise ValueError("end_s precedes start_s")
    fields = raw_label.split(",")
    if len(fields) == 1:
        expected = perceived = fields[0]
        operation = None
    elif len(fields) == 3:
        if source_kind != "manual_annotation":
            raise ValueError("forced alignment cannot contain manual error labels")
        expected, perceived, operation = fields
        if operation not in {"s", "a", "d"}:
            raise ValueError("unsupported annotation operation")
    else:
        raise ValueError("label must contain one or three fields")
    if not _SYMBOL.fullmatch(expected) or not _SYMBOL.fullmatch(perceived):
        raise ValueError("invalid or empty phonetic symbol")
    # Decorated reserved markers have no defined meaning in this contract.
    if any(s.lower() in {"sil*", "err*"} for s in (expected, perceived)):
        raise ValueError("decorated reserved marker")
    e_sil, p_sil = expected.lower() == "sil", perceived.lower() == "sil"
    if operation is not None and expected.lower() == "err":
        raise ValueError("expected side of an error label cannot be err")
    if operation == "a" and (not e_sil or p_sil):
        raise ValueError("addition requires sil -> speech/uncertain")
    if operation == "d" and (e_sil or not p_sil):
        raise ValueError("deletion requires speech -> sil")
    if operation == "s" and (e_sil or p_sil):
        raise ValueError("substitution cannot contain sil")
    if operation == "s" and expected.lower() == perceived.lower():
        raise ValueError("substitution cannot assert identical symbols")
    deviation = any("*" in s for s in (expected, perceived))
    uncertain = perceived.lower() == "err" or deviation
    realized = None if uncertain else perceived
    return {
        "schema_version": "p8.annotation.v0",
        "raw_label": raw_label,
        "expected_phone": expected,
        "expected_phone_normalized": expected.upper(),
        "raw_perceived_symbol": perceived,
        "perceived_symbol_normalized": perceived.upper(),
        "realized_phone": realized,
        "realized_phone_normalized": None if realized is None else realized.upper(),
        "annotation_operation": operation,
        "operation_scope": "source_pronunciation_not_enhancement",
        "source_kind": source_kind,
        "source_id": source_id,
        "annotation_basis": ("manual_source_assertion" if source_kind == "manual_annotation"
                             else "forced_alignment_estimate"),
        "start_s": str(start), "end_s": str(end),
        "raw_start_s": str(start_s), "raw_end_s": str(end_s),
        "start_input_type": type(start_s).__name__,
        "end_input_type": type(end_s).__name__,
        "zero_length": start == end,
        "identity_uncertain": uncertain,
        "deviation": deviation,
        "realized_class": "uncertain" if uncertain else ("silence" if p_sil else "speech"),
        "real_data_approved": False,
        "hard_identity_eligible": False,
    }
