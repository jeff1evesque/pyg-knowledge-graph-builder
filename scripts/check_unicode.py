#!/usr/bin/env python3
"""Reject source containing invisible or homoglyph-prone Unicode.

Three categories, none of which have a legitimate use in this repository:

* **Bidirectional controls** (``U+202A``-``U+202E``, ``U+2066``-``U+2069``).
  These reorder how text is DISPLAYED without changing what the compiler or
  interpreter sees, so reviewed code and executed code can differ. That is the
  Trojan Source class of attack (CVE-2021-42574): a comment can be made to look
  like it ends where it does not, hiding live code inside it.

* **Zero-width and invisible separators** (``U+200B``-``U+200F``, ``U+2060``,
  and ``U+FEFF`` anywhere but the first character). Invisible in every editor
  and diff, so they can hide payloads or silently break string comparisons that
  look identical on screen.

* **Confusable scripts** — Cyrillic and Greek letters outside a small
  allowlist. CYRILLIC SMALL LETTER A (``U+0430``) renders identically to Latin
  ``a`` (``U+0061``) but is a different code point, so two identifiers spelled
  "password" can coexist and only one is the one you think. MICRO SIGN is
  allowed because it is used legitimately for microseconds.

  Note this docstring names those characters rather than showing them: this
  file is scanned by the rule it implements, so illustrating the attack
  literally would make the check fail on itself.

Deliberately NOT rejected: em-dash, en-dash, non-breaking space, accented
Latin, and emoji. Those are used on purpose here — the BLS mappers depend on
``'—'`` and ``'\\u00a0'`` literals to clean scraped tables — and banning them
would generate noise rather than safety.

Only files tracked by git are scanned, and binary files are skipped, so test
fixtures such as Parquet do not produce false positives.

Usage:
    python scripts/check_unicode.py              # scan everything tracked
    python scripts/check_unicode.py FILE...      # scan specific files
    python scripts/check_unicode.py --staged     # scan staged files (hooks)
"""

from __future__ import annotations

import argparse
import subprocess
import unicodedata

#: Reorder displayed text without changing what is parsed.
BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))

#: Invisible; render as nothing in every editor and diff.
ZERO_WIDTH = set(range(0x200B, 0x2010)) | {0x2060, 0x180E}

#: Byte-order mark. Legal as the very first character, suspicious anywhere else.
BOM = 0xFEFF

#: Letters that render like Latin but are not.
CYRILLIC = range(0x0400, 0x0500)
GREEK = range(0x0370, 0x0400)

#: Confusable-script characters with a genuine use here. ``μ`` appears as the
#: micro sign in timing notes ("~1 μs"); both code points render identically.
CONFUSABLE_ALLOWLIST = {0x03BC, 0x00B5}

CATEGORIES = (
    ("bidirectional control", lambda cp: cp in BIDI),
    ("zero-width/invisible", lambda cp: cp in ZERO_WIDTH),
    ("confusable Cyrillic", lambda cp: cp in CYRILLIC),
    ("confusable Greek", lambda cp: cp in GREEK and cp not in CONFUSABLE_ALLOWLIST),
)


def tracked_files(staged: bool) -> list[str]:
    """Files to scan, from git so untracked build output is ignored."""
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def read_text(path: str) -> str | None:
    """Decode a file as UTF-8, or return None if it is binary/unreadable."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (OSError, IsADirectoryError):
        return None
    if b"\x00" in raw:          # NUL byte: binary (Parquet fixtures, images)
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan(path: str, text: str) -> list[str]:
    """Findings for one file, as human-readable lines."""
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            cp = ord(char)
            if cp < 0x80:
                continue
            # A BOM is only acceptable as the first character of the file.
            if cp == BOM and not (lineno == 1 and column == 1):
                findings.append(_report(path, lineno, column, char, "byte-order mark"))
                continue
            for label, matches in CATEGORIES:
                if matches(cp):
                    findings.append(_report(path, lineno, column, char, label))
                    break
    return findings


def _report(path: str, lineno: int, column: int, char: str, label: str) -> str:
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "unnamed control character"
    return f"{path}:{lineno}:{column}: {label} U+{ord(char):04X} ({name})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", help="Files to scan (default: all tracked).")
    parser.add_argument(
        "--staged", action="store_true",
        help="Scan only staged files. Intended for the pre-commit hook.",
    )
    args = parser.parse_args()

    paths = args.paths or tracked_files(args.staged)

    findings: list[str] = []
    scanned = 0
    for path in paths:
        text = read_text(path)
        if text is None:
            continue
        scanned += 1
        findings.extend(scan(path, text))

    if findings:
        print(f"Rejected: {len(findings)} disallowed character(s) "
              f"in {len({f.split(':')[0] for f in findings})} file(s).\n")
        for finding in findings:
            print(f"  {finding}")
        print(
            "\nThese characters are invisible or imitate Latin letters, so what "
            "is reviewed\nand what runs can differ. If one is genuinely needed, "
            "add it to\nCONFUSABLE_ALLOWLIST in scripts/check_unicode.py with a "
            "comment saying why."
        )
        return 1

    print(f"OK: no disallowed Unicode in {scanned} text file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
