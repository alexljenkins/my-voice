"""Word scores match tests/wer.rs. Ties prefer substitution, deletion, insertion."""


def words(text, strict=False):
    if strict:
        return text.split()
    return [word for token in text.lower().split()
            if (word := ''.join(c for c in token if c.isalnum() or c == "'"))]


def score(reference, hypothesis, strict=False):
    a, b = words(reference, strict), words(hypothesis, strict)
    matrix = [list(range(len(b) + 1))]
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(matrix[-1][j - 1] + (x != y), matrix[-1][j] + 1, row[-1] + 1))
        matrix.append(row)
    i, j = len(a), len(b)
    s = d = ins = 0
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            s += a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return dict(substitutions=s, deletions=d, insertions=ins, errors=s + d + ins,
                reference_words=len(a), wer=(s + d + ins) / max(1, len(a)))
