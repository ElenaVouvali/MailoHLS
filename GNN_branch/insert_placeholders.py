import re


def insert_placeholders(cpp_file, allowed_labels=None):
    """Insert directive placeholders only at declared MailoHLS actions."""
    allowed_labels = (
        None
        if allowed_labels is None
        else {
            str(label).strip().upper()
            for label in allowed_labels
        }
    )

    with open(cpp_file, "r", encoding="utf-8") as handle:
        code_lines = handle.readlines()

    output = []
    i = 0

    while i < len(code_lines):
        line = code_lines[i]
        stripped = line.rstrip("\n")

        # /*L1:*/ or L1:
        match = re.match(
            r"^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)",
            stripped,
        )

        if not match:
            output.append(line)
            i += 1
            continue

        label = (match.group(1) or match.group(2)).upper()
        after = stripped[match.end():]

        # Preserve the actual code, but remove an inactive synthetic Lk marker.
        # Otherwise mailohls_contract.py converts it into <SRC_Lk>, falsely
        # advertising it as a prediction/action site.
        if allowed_labels is not None and label not in allowed_labels:
            indentation = line[:len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            output.append(
                f"{indentation}{after.lstrip()}{newline}"
            )
            i += 1
            continue

        output.append(line)

        if re.search(r"\bfor\s*\(", after):
            output.append(
                f"#pragma HLS pipeline II=auto{{_PIPE_{label}}}\n"
            )
            output.append(
                f"#pragma HLS unroll factor=auto{{_UNROLL_{label}}}\n"
            )
            i += 1
            continue

        array_match = re.search(
            r"^(?:\s*[A-Za-z_]\w*\s*:\s*)?"
            r"\s*([A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*)"
            r"\s+([A-Za-z_]\w*)"
            r"\s*(\[[^\]]+\])+"
            r"(?:\s*=\s*[^;]+)?"
            r"\s*;",
            after,
        )

        if array_match:
            variable_name = array_match.group(2)
            output.append(
                f"#pragma HLS array_partition "
                f"variable={variable_name} "
                f"type=auto{{_ARRAY_T_{label}}} "
                f"factor=auto{{_ARRAY_F_{label}}} "
                f"dim=auto{{_ARRAY_D_{label}}}\n"
            )

        i += 1

    return output
