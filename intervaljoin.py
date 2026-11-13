"""Correct reference implementation; intentionally rebuilds the full join."""


def _validated(upserts, deletes):
    if not isinstance(upserts, list) or not isinstance(deletes, list):
        raise ValueError('lists required')
    rows, seen = {}, set()
    for row in upserts:
        if (not isinstance(row, list) or len(row) != 4
                or not all(isinstance(x, str) and x for x in row[:2])
                or type(row[2]) is not int or type(row[3]) is not int or row[2] >= row[3]):
            raise ValueError('invalid interval')
        if row[0] in rows:
            raise ValueError('duplicate upsert')
        rows[row[0]] = tuple(row)
    for ident in deletes:
        if not isinstance(ident, str) or not ident or ident in seen or ident in rows:
            raise ValueError('invalid delete')
        seen.add(ident)
    return rows, seen


def _sort_key(row):
    key, left, right, start, end = row
    return key, (left is not None, left or ''), (right is not None, right or ''), start, end


def _gaps(start, end, cover):
    cursor, result = start, []
    for lo, hi in sorted(cover):
        lo, hi = max(start, lo), min(end, hi)
        if lo >= hi:
            continue
        if cursor < lo:
            result.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < end:
        result.append((cursor, end))
    return result


def _rebuild(left, right, touch):
    rows = set()
    right_cover = {ident: [] for ident in right}
    for lid, lrow in left.items():
        touch('left', 1)
        _, key, start, end = lrow
        cover = []
        for rid, rrow in right.items():
            touch('right', 1)
            if key != rrow[1]:
                continue
            lo, hi = max(start, rrow[2]), min(end, rrow[3])
            if lo < hi:
                rows.add((key, lid, rid, lo, hi))
                cover.append((lo, hi))
                right_cover[rid].append((lo, hi))
        for lo, hi in _gaps(start, end, cover):
            rows.add((key, lid, None, lo, hi))
    for rid, rrow in right.items():
        touch('right', 1)
        _, key, start, end = rrow
        for lo, hi in _gaps(start, end, right_cover[rid]):
            rows.add((key, None, rid, lo, hi))
    return rows


class Engine:
    def __init__(self, calibration):
        self._right, _ = _validated(calibration, [])
        self._left = {}
        self._work = {'left_visits': 0, 'right_visits': 0}
        self._rows = _rebuild(self._left, self._right, self._touch)
        self._work = {'left_visits': 0, 'right_visits': 0}

    def _touch(self, side, count=1):
        self._work[side + '_visits'] += count

    def apply(self, upserts, deletes):
        changed, removed = _validated(upserts, deletes)
        proposed = dict(self._left)
        for ident in removed:
            proposed.pop(ident, None)
        proposed.update(changed)
        self._work = {'left_visits': 0, 'right_visits': 0}
        after = _rebuild(proposed, self._right, self._touch)
        added, lost = after - self._rows, self._rows - after
        self._left, self._rows = proposed, after
        return {'add': [list(row) for row in sorted(added, key=_sort_key)],
                'remove': [list(row) for row in sorted(lost, key=_sort_key)]}

    def read(self):
        return [list(row) for row in sorted(self._rows, key=_sort_key)]

    def work(self):
        return dict(self._work)
