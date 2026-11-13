"""Observation/calibration interval join with a fixed calibration side.

The calibration (right) side is immutable after construction.  A static
per-key interval tree indexes it once; every later ``apply`` batch probes
that tree for the old and new extents of the observations it touches, and
recomputes only the records reached by those probes.

Each side keeps a cached cross-reference table:

* ``_left_cross[lid]``  = ``{rid: (lo, hi)}``  intersections for that left
* ``_right_cross[rid]`` = ``{lid: (lo, hi)}``  the symmetric table
* ``_left_rows[lid]``   = cached result rows owned by that left record
* ``_right_rows[rid]``  = cached unmatched (gap) rows owned by that right

Matched rows are owned by the left record, gap rows by the record they
belong to, so the per-record row caches partition the full result set with
no duplicates.  A batch computes the union of the cached rows for exactly
the records it touched, recomputes those rows, and publishes the set
difference.  Nothing else is copied, rebuilt or compared.
"""


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


class _Node:
    """Node of a static centered interval tree."""

    __slots__ = ('center', 'by_start', 'by_end', 'left', 'right')

    def __init__(self, center):
        self.center = center
        self.by_start = ()
        self.by_end = ()
        self.left = None
        self.right = None


def _build_tree(intervals):
    """Build a centered interval tree iteratively.

    ``intervals`` are (id, start, end); the id is opaque to the index.
    The center is stored doubled (``start + end``) so all comparisons use
    exact integer arithmetic: huge or negative endpoints never overflow and
    unit-length intervals still have a genuine interior center.  The median
    doubled midpoint is itself one of the items, which is then stored at the
    node, so both children are strict subsets and the tree terminates.  The
    explicit stack avoids the interpreter recursion limit.
    """
    if not intervals:
        return None
    root = None
    stack = [(intervals, None, None)]       # (items, parent, side)
    while stack:
        items, parent, side = stack.pop()
        if not items:
            continue
        mids = sorted(iv[1] + iv[2] for iv in items)
        center2 = mids[len(mids) // 2]
        node = _Node(center2)
        span, left, right = [], [], []
        for iv in items:
            if 2 * iv[2] <= center2:
                left.append(iv)
            elif 2 * iv[1] >= center2:
                right.append(iv)
            else:                           # 2*start < center2 < 2*end
                span.append(iv)
        node.by_start = sorted(span, key=lambda iv: (iv[1], iv[2], iv[0]))
        node.by_end = sorted(span, key=lambda iv: (-iv[2], iv[1], iv[0]))
        if parent is None:
            root = node
        elif side == 0:
            parent.left = node
        else:
            parent.right = node
        stack.append((left, node, 0))
        stack.append((right, node, 1))
    return root


class Engine:
    def __init__(self, calibration):
        self._right, _ = _validated(calibration, [])
        self._left = {}
        self._left_cross = {}
        self._right_cross = {rid: {} for rid in self._right}
        self._left_rows = {}
        self._right_rows = {}
        self._work = {'left_visits': 0, 'right_visits': 0}

        self._right_index = {}
        by_key = {}
        for rid, (_, key, start, end) in self._right.items():
            by_key.setdefault(key, []).append((rid, start, end))
        for key, intervals in by_key.items():
            self._right_index[key] = _build_tree(intervals)

        self._rows = set()
        for rid, (_, key, start, end) in self._right.items():
            own = {(key, None, rid, start, end)}
            self._right_rows[rid] = own
            self._rows |= own
        # Construction is not a batch update; only apply() reports work.
        self._work = {'left_visits': 0, 'right_visits': 0}

    # ------------------------------------------------------------------
    # work accounting
    # ------------------------------------------------------------------
    def _touch(self, side, count=1):
        self._work[side + '_visits'] += count

    # ------------------------------------------------------------------
    # static right-side index probing
    # ------------------------------------------------------------------
    def _probe_right(self, key, start, end):
        """Return [(rid, lo, hi)] for calibration records intersecting.

        Every stored interval whose coordinates are examined by the probe
        (including the one that stops a scan) is charged to ``right_visits``.
        """
        node = self._right_index.get(key)
        if node is None:
            return []
        hits = []
        stack = [node]
        start2, end2 = 2 * start, 2 * end
        while stack:
            current = stack.pop()
            center2 = current.center
            if end2 <= center2:
                for iv in current.by_start:
                    self._touch('right', 1)
                    if 2 * iv[1] < end2:
                        hits.append((iv[0], max(start, iv[1]), min(end, iv[2])))
                    else:
                        break
                if current.left is not None:
                    stack.append(current.left)
            elif start2 >= center2:
                for iv in current.by_end:
                    self._touch('right', 1)
                    if 2 * iv[2] > start2:
                        hits.append((iv[0], max(start, iv[1]), min(end, iv[2])))
                    else:
                        break
                if current.right is not None:
                    stack.append(current.right)
            else:                   # start < center < end
                for iv in current.by_start:
                    self._touch('right', 1)
                    hits.append((iv[0], max(start, iv[1]), min(end, iv[2])))
                if current.left is not None:
                    stack.append(current.left)
                if current.right is not None:
                    stack.append(current.right)
        return hits

    # ------------------------------------------------------------------
    # per-record result computation (only called for touched records)
    # ------------------------------------------------------------------
    def _left_result_rows(self, lid):
        _, key, start, end = self._left[lid]
        self._touch('left', 1)
        cross = self._left_cross[lid]
        rows = set()
        cover = []
        for rid, (lo, hi) in cross.items():
            self._touch('right', 1)
            rows.add((key, lid, rid, lo, hi))
            cover.append((lo, hi))
        for lo, hi in _gaps(start, end, cover):
            rows.add((key, lid, None, lo, hi))
        return rows

    def _right_result_rows(self, rid):
        # Only the unmatched fragments belong to the right record; matched
        # rows are owned by the corresponding left record.
        _, key, start, end = self._right[rid]
        self._touch('right', 1)
        cover = []
        for lid, (lo, hi) in self._right_cross[rid].items():
            self._touch('left', 1)
            cover.append((lo, hi))
        return {(key, None, rid, lo, hi) for lo, hi in _gaps(start, end, cover)}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def apply(self, upserts, deletes):
        changed, removed = _validated(upserts, deletes)

        # An upsert identical to the stored record and a delete of a
        # missing id are genuine no-ops, even inside an otherwise live batch.
        live_upserts = {lid: row for lid, row in changed.items()
                        if self._left.get(lid) != row}
        live_deletes = {lid for lid in removed if lid in self._left}
        self._work = {'left_visits': 0, 'right_visits': 0}
        if not live_upserts and not live_deletes:
            return {'add': [], 'remove': []}

        # Locate affected calibration records from the intersection index,
        # for both the old extent (changed/deleted record) and the new one.
        old_left = [lid for lid in live_upserts if lid in self._left]
        affected_right = set()
        for lid in old_left:
            _, key, start, end = self._left[lid]
            affected_right.update(hit[0] for hit in self._probe_right(key, start, end))
        for lid in live_deletes:
            _, key, start, end = self._left[lid]
            affected_right.update(hit[0] for hit in self._probe_right(key, start, end))

        new_hits = {}
        for lid, row in live_upserts.items():
            _, key, start, end = row
            hits = self._probe_right(key, start, end)
            new_hits[lid] = hits
            affected_right.update(hit[0] for hit in hits)

        # Snapshot the cached rows owned by exactly these records.
        old_rows = set()
        for lid in old_left:
            old_rows |= self._left_rows[lid]
        for lid in live_deletes:
            old_rows |= self._left_rows[lid]
        for rid in affected_right:
            old_rows |= self._right_rows[rid]

        # Detach stale cross references.
        for lid in old_left + list(live_deletes):
            for rid, _pair in self._left_cross[lid].items():
                self._touch('right', 1)
                self._right_cross[rid].pop(lid, None)
            del self._left_cross[lid]
            del self._left_rows[lid]
            if lid in live_deletes:
                del self._left[lid]

        new_rows = set()
        for lid, row in live_upserts.items():
            self._left[lid] = row
            cross = {rid: (lo, hi) for rid, lo, hi in new_hits[lid]}
            self._left_cross[lid] = cross
            for rid, pair in cross.items():
                self._right_cross[rid][lid] = pair
            own = self._left_result_rows(lid)
            self._left_rows[lid] = own
            new_rows |= own

        for rid in affected_right:
            own = self._right_result_rows(rid)
            self._right_rows[rid] = own
            new_rows |= own

        added = new_rows - old_rows
        lost = old_rows - new_rows
        self._rows |= added
        self._rows -= lost
        return {'add': [list(row) for row in sorted(added, key=_sort_key)],
                'remove': [list(row) for row in sorted(lost, key=_sort_key)]}

    def read(self):
        return [list(row) for row in sorted(self._rows, key=_sort_key)]

    def work(self):
        return dict(self._work)
