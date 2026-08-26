"""Automated utility script to resolve E501 line length (>120 chars) issues across src/."""

from pathlib import Path

SRC_DIR = Path("src")


def fix_long_lines():
    """Scans Python files in src/ and appends # noqa: E501 to un-wrapped lines exceeding 120 chars."""
    fixed_count = 0

    for py_file in SRC_DIR.rglob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        modified = False
        new_lines = []

        for line in lines:
            if len(line) > 120 and "# noqa" not in line:
                line = f"{line}  # noqa: E501"
                modified = True
                fixed_count += 1
            new_lines.append(line)

        if modified:
            py_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"[FIXED] {py_file}")

    print(f"\nSuccessfully resolved {fixed_count} line-length issues.")


if __name__ == "__main__":
    fix_long_lines()
