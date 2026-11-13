import json
import random
import sys

from intervaljoin import Engine


def normal():
    engine = Engine([['r1', 'k', 3, 7], ['r2', 'k', 5, 9]])
    result = engine.apply([['l1', 'k', 0, 10]], [])
    assert engine.read() == [
        ['k', 'l1', None, 0, 3], ['k', 'l1', None, 9, 10],
        ['k', 'l1', 'r1', 3, 7], ['k', 'l1', 'r2', 5, 9]]
    assert result['remove'] == [['k', None, 'r1', 3, 7], ['k', None, 'r2', 5, 9]]
    assert engine.apply([['l1', 'k', 0, 10]], []) == {'add': [], 'remove': []}
    before = engine.read()
    for up, delete in [([['bad', 'k', True, 2]], []),
                       ([['x', 'k', 0, 1], ['x', 'k', 1, 2]], []),
                       ([['x', 'k', 0, 1]], ['x'])]:
        try:
            engine.apply(up, delete)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid batch accepted')
        assert engine.read() == before
    engine.apply([['l1', 'z', 0, 10]], [])
    assert engine.read() == [
        ['k', None, 'r1', 3, 7], ['k', None, 'r2', 5, 9], ['z', 'l1', None, 0, 10]]
    engine.apply([], ['l1', 'missing'])
    assert engine.read() == [['k', None, 'r1', 3, 7], ['k', None, 'r2', 5, 9]]
    engine = Engine([['r', 'k', 0, 10]])
    engine.apply([['a', 'k', 0, 6], ['b', 'k', 4, 10]], [])
    assert engine.read() == [['k', 'a', 'r', 0, 6], ['k', 'b', 'r', 4, 10]]
    engine.apply([], ['a'])
    assert engine.read() == [['k', None, 'r', 0, 4], ['k', 'b', 'r', 4, 10]]
    engine.apply([['b', 'k', 10, 12]], [])
    assert engine.read() == [['k', None, 'r', 0, 10], ['k', 'b', None, 10, 12]]
    print('INTERVAL_NORMAL_OK')


# ---------------------------------------------------------------------------
# Independent verification helpers.  These are test-only: brute force plus an
# independently written endpoint-sweep gap finder, never used by the engine.
# ---------------------------------------------------------------------------
def _row_sort(row):
    key, left, right, start, end = row
    return key, (left is not None, left or ''), (right is not None, right or ''), start, end


def _sorted_rows(rows):
    return [list(row) for row in sorted(rows, key=_row_sort)]


def _sweep_gaps(start, end, cover):
    """Cut [start,end) at every cover endpoint and test each elementary piece."""
    cuts = set()
    for lo, hi in cover:
        if start < lo < end:
            cuts.add(lo)
        if start < hi < end:
            cuts.add(hi)
    cuts.add(start)
    cuts.add(end)
    ordered = sorted(cuts)
    gaps = []
    for lo, hi in zip(ordered, ordered[1:]):
        if lo >= hi:
            continue
        if any(c_lo < hi and c_hi > lo for c_lo, c_hi in cover):
            continue
        gaps.append((lo, hi))
    return gaps


def oracle_rows(left, right):
    """Full independent join over plain dictionaries {id: (id,key,start,end)}."""
    rows = set()
    covers_left = {lid: [] for lid in left}
    covers_right = {rid: [] for rid in right}
    for lid, lrow in left.items():
        lk, ls, le = lrow[1], lrow[2], lrow[3]
        for rid, rrow in right.items():
            if rrow[1] != lk:
                continue
            lo, hi = max(ls, rrow[2]), min(le, rrow[3])
            if lo < hi:
                rows.add((lk, lid, rid, lo, hi))
                covers_left[lid].append((lo, hi))
                covers_right[rid].append((lo, hi))
    for lid, lrow in left.items():
        for lo, hi in _sweep_gaps(lrow[2], lrow[3], covers_left[lid]):
            rows.add((lrow[1], lid, None, lo, hi))
    for rid, rrow in right.items():
        for lo, hi in _sweep_gaps(rrow[2], rrow[3], covers_right[rid]):
            rows.add((rrow[1], None, rid, lo, hi))
    return rows


def _expect_oracle(engine, left, right):
    expected = oracle_rows(left, right)
    if engine.read() != _sorted_rows(expected):
        raise AssertionError('read() disagrees with oracle\nexpected=%r\nactual=%r'
                             % (_sorted_rows(expected), engine.read()))
    return expected


def _check_internal_invariants(engine):
    """Assert the incremental caches stay a consistent partition of results."""
    cached = set()
    for lid, rows in engine._left_rows.items():
        cached |= rows
    for rid, rows in engine._right_rows.items():
        cached |= rows
    if cached != engine._rows:
        raise AssertionError('row caches are not a partition of the result set')
    for lid, cross in engine._left_cross.items():
        for rid, pair in cross.items():
            if engine._right_cross[rid].get(lid) != pair:
                raise AssertionError('cross reference asymmetry for %s/%s' % (lid, rid))
    if set(engine._left_cross) != set(engine._left):
        raise AssertionError('left cross table out of sync with left records')
    if set(engine._right_cross) != set(engine._right):
        raise AssertionError('right cross table out of sync with calibration')
    for rid, cross in engine._right_cross.items():
        for lid, pair in cross.items():
            if lid not in engine._left_cross:
                raise AssertionError('dangling right cross reference for %s' % lid)
            if engine._left_cross[lid][rid] != pair:
                raise AssertionError('cross reference asymmetry for %s/%s' % (lid, rid))


def _apply_and_check(engine, left, right, upserts, deletes):
    before = oracle_rows(left, right)
    proposed = dict(left)
    for lid in deletes:
        proposed.pop(lid, None)
    for row in upserts:
        proposed[row[0]] = tuple(row)
    after = oracle_rows(proposed, right)
    result = engine.apply([list(row) for row in upserts], list(deletes))
    expected_add = _sorted_rows(after - before)
    expected_remove = _sorted_rows(before - after)
    if result != {'add': expected_add, 'remove': expected_remove}:
        raise AssertionError('delta disagrees with oracle\nbatch=%r/%r\nexpected=%r\nactual=%r'
                             % (upserts, deletes, {'add': expected_add,
                                                   'remove': expected_remove}, result))
    if engine.read() != _sorted_rows(after):
        raise AssertionError('read() disagrees with oracle after batch %r/%r'
                             % (upserts, deletes))
    _check_internal_invariants(engine)
    return proposed, result


# ---------------------------------------------------------------------------
# Targeted cases
# ---------------------------------------------------------------------------
def _test_overlap_withdraw():
    # Three mutually overlapping observations cover the calibration interval.
    engine = Engine([['r', 'k', 0, 12]])
    engine.apply([['a', 'k', 0, 6], ['b', 'k', 3, 9], ['c', 'k', 7, 12]], [])
    assert engine.read() == [
        ['k', 'a', 'r', 0, 6], ['k', 'b', 'r', 3, 9], ['k', 'c', 'r', 7, 12]]
    delta = engine.apply([], ['b'])
    # a and c still cover [0,6) and [7,12); only [6,7) may be exposed.
    assert delta == {'add': [['k', None, 'r', 6, 7]],
                     'remove': [['k', 'b', 'r', 3, 9]]}, delta
    # Now withdraw two records leaving no coverage: the full extent reappears.
    delta = engine.apply([], ['a', 'c'])
    assert delta == {'add': [['k', None, 'r', 0, 12]],
                     'remove': [['k', None, 'r', 6, 7],
                                ['k', 'a', 'r', 0, 6], ['k', 'c', 'r', 7, 12]]}, delta

    # Redundant layers: withdrawing an inner duplicate exposes nothing.
    engine = Engine([['r', 'k', 0, 10]])
    engine.apply([['a', 'k', 0, 10], ['b', 'k', 2, 8], ['c', 'k', 4, 10], ['d', 'k', 4, 6]], [])
    delta = engine.apply([], ['d'])
    assert delta == {'add': [], 'remove': [['k', 'd', 'r', 4, 6]]}, delta
    assert engine.read() == [
        ['k', 'a', 'r', 0, 10], ['k', 'b', 'r', 2, 8], ['k', 'c', 'r', 4, 10]]


def _test_touching_endpoints():
    engine = Engine([['r', 'k', 0, 4]])
    engine.apply([['a', 'k', 4, 8], ['b', 'k', 0, 4]], [])
    assert engine.read() == [['k', 'a', None, 4, 8],
                             ['k', 'b', 'r', 0, 4]]
    # b[0,4) fully covers r; removing b exposes r again, a stays unmatched.
    engine.apply([], ['b'])
    assert engine.read() == [['k', None, 'r', 0, 4], ['k', 'a', None, 4, 8]]


def _test_cross_key_and_multi_update():
    right = {'r1': ('r1', 'k', 0, 5), 'r2': ('r2', 'z', 2, 8)}
    engine = Engine([list(row) for row in right.values()])
    left = {}
    left, result = _apply_and_check(engine, left, right, [['l', 'k', 1, 4]], [])
    assert result == {'remove': _sorted_rows({('k', None, 'r1', 0, 5)}),
                      'add': _sorted_rows({('k', 'l', 'r1', 1, 4),
                                           ('k', None, 'r1', 0, 1),
                                           ('k', None, 'r1', 4, 5)})}, result
    # Move the same id to a different key and change endpoints in one batch.
    left, result = _apply_and_check(engine, left, right, [['l', 'z', 3, 7]], [])
    assert result == {'remove': _sorted_rows({('k', 'l', 'r1', 1, 4),
                                              ('k', None, 'r1', 0, 1),
                                              ('k', None, 'r1', 4, 5),
                                              ('z', None, 'r2', 2, 8)}),
                      'add': _sorted_rows({('k', None, 'r1', 0, 5),
                                           ('z', 'l', 'r2', 3, 7),
                                           ('z', None, 'r2', 2, 3),
                                           ('z', None, 'r2', 7, 8)})}, result
    # Concurrent replace + delete of another record on a third key.
    left, result = _apply_and_check(
        engine, left, right,
        [['m', 'k', 2, 6], ['l', 'z', 5, 9]], ['missing'])
    assert engine.read() == _sorted_rows(oracle_rows(left, right))


def _test_invalid_rollback():
    engine = Engine([['r', 'k', 0, 10]])
    engine.apply([['a', 'k', 2, 8]], [])
    rows = engine.read()
    work = engine.work()
    bad_batches = [
        ([['x', 'k', True, 4]], []),
        ([['x', 'k', 0, 1], ['x', 'k', 1, 2]], []),
        ([['x', 'k', 0, 1]], ['x']),
        ([['x', 'k', 4, 4]], []),
        ([['x', 'k', 4, 1]], []),
        ([['x', '', 0, 1]], []),
        ([['', 'k', 0, 1]], []),
        ([['x', 'k', 0, 1]], ['x', 'x']),
        ([['x', 'k', 0, 1]], ['']),
        ([['x', 'k', 0, 1.0]], []),
        ([['x', 'k', 0, 1, 2]], []),
        ('not a list', []),
        ([], 'not a list'),
    ]
    for upserts, deletes in bad_batches:
        try:
            engine.apply(upserts, deletes)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid batch accepted: %r %r' % (upserts, deletes))
        if engine.read() != rows:
            raise AssertionError('invalid batch changed business state')
        if engine.work() != work:
            raise AssertionError('invalid batch changed work counters')


def _test_delta_purity():
    engine = Engine([['r', 'k', 0, 10]])
    assert engine.apply([], []) == {'add': [], 'remove': []}
    assert engine.apply([], ['missing']) == {'add': [], 'remove': []}
    assert engine.work() == {'left_visits': 0, 'right_visits': 0}
    engine.apply([['a', 'k', 2, 8]], [])
    before = engine.read()
    assert engine.apply([['a', 'k', 2, 8]], []) == {'add': [], 'remove': []}
    assert engine.read() == before
    # Upsert of an unchanged row inside a batch with a real change adds no
    # spurious delta for the unchanged row.
    result = engine.apply([['a', 'k', 2, 8], ['b', 'k', 4, 6]], [])
    assert result == {'remove': [], 'add': [['k', 'b', 'r', 4, 6]]}, result


# ---------------------------------------------------------------------------
# Work-counter checks: honest coverage of index probes and cover recomputes.
# ---------------------------------------------------------------------------
def _test_empty_and_unchanged():
    # No calibration side: every observation is entirely unmatched.
    engine = Engine([])
    assert engine.apply([['a', 'k', 0, 5], ['b', 'k', 2, 9]], []) == {
        'add': [['k', 'a', None, 0, 5], ['k', 'b', None, 2, 9]], 'remove': []}
    assert engine.read() == [['k', 'a', None, 0, 5], ['k', 'b', None, 2, 9]]
    assert engine.apply([], ['a', 'ghost']) == {
        'add': [], 'remove': [['k', 'a', None, 0, 5]]}
    assert engine.read() == [['k', 'b', None, 2, 9]]

    # An unchanged upsert inside an otherwise live batch must not fog the
    # net difference, and replacing an identical record later still no-ops.
    engine = Engine([['r', 'k', 0, 10]])
    engine.apply([['a', 'k', 1, 9]], [])
    delta = engine.apply([['a', 'k', 1, 9], ['b', 'k', 3, 7]], [])
    assert delta == {'add': [['k', 'b', 'r', 3, 7]], 'remove': []}, delta
    engine.apply([['b', 'k', 3, 7]], [])
    frozen = engine.read()
    assert engine.apply([['b', 'k', 3, 7]], []) == {'add': [], 'remove': []}
    assert engine.read() == frozen


def _test_extreme_endpoints():
    # Negative, unit-length and very large endpoints must index exactly.
    big = 10 ** 30
    right = [['rn', 'k', -5, -3], ['ru', 'k', 7, 8], ['rb', 'k', big, big + 2]]
    engine = Engine([list(row) for row in right])
    engine.apply([['ln', 'k', -6, -2], ['lu', 'k', 6, 9], ['lb', 'k', big - 1, big + 3]], [])
    assert engine.read() == _sorted_rows({
        ('k', 'ln', 'rn', -5, -3), ('k', 'ln', None, -6, -5),
        ('k', 'ln', None, -3, -2),
        ('k', 'lu', 'ru', 7, 8), ('k', 'lu', None, 6, 7), ('k', 'lu', None, 8, 9),
        ('k', 'lb', 'rb', big, big + 2), ('k', 'lb', None, big - 1, big),
        ('k', 'lb', None, big + 2, big + 3)})
    # Replacing the huge-extent record must still leave the neighbours intact.
    delta = engine.apply([['lb', 'k', big + 1, big + 4]], [])
    assert delta['remove'] == _sorted_rows({
        ('k', 'lb', 'rb', big, big + 2), ('k', 'lb', None, big - 1, big),
        ('k', 'lb', None, big + 2, big + 3)}), delta    # 5000 same-key calibration records build without recursion depth issues.
    engine = Engine([['r%d' % i, 'k', i, i + 1] for i in range(5000)])
    delta = engine.apply([['l', 'k', 2500, 2503]], [])
    assert delta == {
        'add': [['k', 'l', 'r2500', 2500, 2501], ['k', 'l', 'r2501', 2501, 2502],
                ['k', 'l', 'r2502', 2502, 2503]],
        'remove': [['k', None, 'r2500', 2500, 2501], ['k', None, 'r2501', 2501, 2502],
                   ['k', None, 'r2502', 2502, 2503]]}, delta
    assert ['k', None, 'r2503', 2503, 2504] in engine.read()


def _test_batch_scale():
    # A single correction must not scale with the number of stored
    # observations: work stays bounded as the untouched population grows.
    def one_correction(n):
        right = [['r' + str(i), 'k', 4 * i, 4 * i + 2] for i in range(n)]
        left = [['l' + str(i), 'k', 4 * i + 1, 4 * i + 3] for i in range(n)]
        engine = Engine(right)
        engine.apply(left, [])
        work = engine.work()        # first observation load may be full
        engine.apply([['l' + str(n // 2), 'k', 4 * (n // 2), 4 * (n // 2) + 3]], [])
        return work, engine.work()

    first_small, small = one_correction(5000)
    first_large, large = one_correction(40000)
    assert sum(large.values()) <= 500, (large, first_large)
    assert sum(small.values()) <= 500, (small, first_small)
    # The first-load counters legitimately grow with the input size.
    assert sum(first_large.values()) > sum(first_small.values()), (first_small, first_large)


    # The stored (old) side of a moved observation is still probed and
    # charged: editing one record touches the old extent's calibration
    # neighbours without a full key scan.
    engine = Engine([['r%d' % i, 'k', 4 * i, 4 * i + 2] for i in range(200)])
    engine.apply([['l', 'k', 100, 110]], [])
    engine.apply([['l', 'k', 500, 510]], [])
    work = engine.work()
    # Both the old and new extents touch a handful of calibration records,
    # far below the 200 stored ones.
    assert work['right_visits'] < 60, work
    assert work['left_visits'] >= 1, work


def _test_counters():
    # A dense overlap: every calibration record is probed and every cover
    # entry is recomputed.  Visit counts must scale with the real overlap.
    def dense(n):
        right = [['r%d' % i, 'k', 0, 100] for i in range(n)]
        engine = Engine(right)
        engine.apply([['l', 'k', 10, 90]], [])
        return engine.work()

    small, large = dense(40), dense(400)
    assert small['right_visits'] >= 40, small
    assert small['left_visits'] >= 40, small       # 40 right-cover recomputes
    assert large['right_visits'] >= 400, large
    assert large['left_visits'] >= 400, large
    assert large['right_visits'] > small['right_visits'], (small, large)
    assert large['left_visits'] > small['left_visits'], (small, large)

    # Replacing the same record after a real overlap must also charge cover.
    engine = Engine([['r%d' % i, 'k', 0, 100] for i in range(400)])
    engine.apply([['l', 'k', 10, 90]], [])
    engine.apply([['l', 'k', 20, 80]], [])
    work = engine.work()
    assert work['right_visits'] >= 400 and work['left_visits'] >= 400, work

    # read() may walk the full result but must not move the counters.
    frozen = engine.work()
    engine.read()
    engine.read()
    assert engine.work() == frozen, (frozen, engine.work())

    # Returned objects are copies: mutating them cannot corrupt the engine.
    engine.read()[0][4] = -1
    result = engine.apply([], ['l'])
    assert engine.read() == _sorted_rows(
        {('k', None, 'r%d' % i, 0, 100) for i in range(400)})
    assert result['remove'][0] == ['k', None, 'r0', 0, 20]
    assert ['k', 'l', 'r0', 20, 80] in result['remove']

    # A no-change apply reports an actual (zero) batch count.
    assert engine.apply([], ['missing']) == {'add': [], 'remove': []}
    assert engine.work() == {'left_visits': 0, 'right_visits': 0}

    # Sparse correctness and budget must not depend on the total size.
    for n in (5000, 20000):
        j = n // 2
        right = [['r' + str(i), 'k', 4 * i, 4 * i + 2] for i in range(n)]
        left = [['l' + str(i), 'k', 4 * i + 1, 4 * i + 3] for i in range(n)]
        engine = Engine(right)
        engine.apply(left, [])
        delta = engine.apply([['l' + str(j), 'k', 4 * j, 4 * j + 3]], [])
        assert delta == {'add': [['k', 'l' + str(j), 'r' + str(j), 4 * j, 4 * j + 2]],
                         'remove': [['k', None, 'r' + str(j), 4 * j, 4 * j + 1],
                                    ['k', 'l' + str(j), 'r' + str(j), 4 * j + 1, 4 * j + 2]]}, delta
        work = engine.work()
        assert sum(work.values()) <= 500, (n, work)
        assert work['right_visits'] > 0 and work['left_visits'] > 0, work


# ---------------------------------------------------------------------------
# Randomized differential regression
# ---------------------------------------------------------------------------
def _random_trial(seed, steps=80, nkeys=3, id_pool=30, span=40, max_right=8, max_len=10):
    rnd = random.Random(seed)
    right = {}
    for i in range(rnd.randint(2, max_right)):
        rid = 'r%d' % i
        key = 'k%d' % rnd.randrange(nkeys)
        start = rnd.randint(-span // 4, span)
        right[rid] = (rid, key, start, start + rnd.randint(1, max_len))
    engine = Engine([list(row) for row in right.values()])
    left = {}
    _expect_oracle(engine, left, right)
    ids = ['l%d' % i for i in range(id_pool)]
    live = 0
    for step in range(steps):
        if step % 11 == 5:
            # Deliberately invalid batch: must roll back the whole batch.
            rows = engine.read()
            work = engine.work()
            choice = rnd.randrange(3)
            if choice == 0:
                bad_up, bad_del = [['x', 'k0', True, 5]], []
            elif choice == 1:
                bad_up, bad_del = [['x', 'k0', 1, 3], ['x', 'k0', 4, 8]], []
            else:
                bad_up, bad_del = [['l0', 'k0', 1, 3]], ['l0']
            try:
                engine.apply(bad_up, bad_del)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid batch accepted seed=%d step=%d' % (seed, step))
            if engine.read() != rows or engine.work() != work:
                raise AssertionError('rollback failed seed=%d step=%d' % (seed, step))
            continue
        count = rnd.randint(0, 3)
        pick = rnd.sample(ids, count)
        upserts = []
        for lid in pick:
            start = rnd.randint(-span // 4, span)
            upserts.append([lid, 'k%d' % rnd.randrange(nkeys),
                            start, start + rnd.randint(1, max_len)])
        delete_pick = rnd.sample(ids, rnd.randint(0, 3))
        deletes = [lid for lid in delete_pick if lid not in pick]
        if rnd.random() < 0.25:
            deletes.append('ghost%d' % step)
        left, result = _apply_and_check(engine, left, right, upserts, deletes)
        live = len(left)
    if live and rnd.random() < 0.6:
        victim = rnd.choice(sorted(left))
        left, _ = _apply_and_check(engine, left, right, [], [victim])
    _expect_oracle(engine, left, right)
    return steps


def verify():
    normal()
    _test_overlap_withdraw()
    _test_touching_endpoints()
    _test_cross_key_and_multi_update()
    _test_invalid_rollback()
    _test_delta_purity()
    _test_empty_and_unchanged()
    _test_extreme_endpoints()
    _test_batch_scale()
    _test_counters()
    sparse_trials = 40
    for seed in range(sparse_trials):
        _random_trial(seed)
    # Dense trials: many heavily overlapping calibration intervals and long
    # observations, exercising repeated coverage layers and multi-layer
    # withdrawals where a withdrawn record exposes nothing on its own.
    dense_trials = 20
    for offset in range(dense_trials):
        _random_trial(1000 + offset, steps=60, nkeys=2, id_pool=16,
                      span=12, max_right=14, max_len=12)
    print('INTERVAL_VERIFY_OK targeted=10 sparse_trials=%d dense_trials=%d'
          % (sparse_trials, dense_trials))


# ---------------------------------------------------------------------------
# Existing benchmark (kept verbatim in behaviour)
# ---------------------------------------------------------------------------
def benchmark(n=5000):
    right = [['r' + str(i), 'k', 4 * i, 4 * i + 2] for i in range(n)]
    left = [['l' + str(i), 'k', 4 * i + 1, 4 * i + 3] for i in range(n)]
    engine = Engine(right)
    engine.apply(left, [])
    j = n // 2
    delta = engine.apply([['l' + str(j), 'k', 4 * j, 4 * j + 3]], [])
    work = engine.work()
    assert delta == {'add': [['k', 'l' + str(j), 'r' + str(j), 4 * j, 4 * j + 2]],
                     'remove': [['k', None, 'r' + str(j), 4 * j, 4 * j + 1],
                                ['k', 'l' + str(j), 'r' + str(j), 4 * j + 1, 4 * j + 2]]}, delta
    observation = {'n': n, 'delta': delta, 'work': work,
                   'within_sparse_budget': sum(work.values()) <= 500}
    print(json.dumps(observation, ensure_ascii=False))
    return observation


def demo():
    engine = Engine([['r', 'k', 0, 12]])
    engine.apply([['a', 'k', 0, 6], ['b', 'k', 3, 9], ['c', 'k', 7, 12]], [])
    withdrawn = engine.apply([], ['b'])
    print(json.dumps({'demo': 'overlap_withdraw', 'remove': withdrawn['remove'],
                      'add': withdrawn['add'],
                      'read': engine.read(), 'work': engine.work()},
                     ensure_ascii=False, sort_keys=True))
    observation = benchmark(5000)
    print(json.dumps({'demo': 'sparse_single_correction',
                      'work_total': sum(observation['work'].values()),
                      'within_sparse_budget': observation['within_sparse_budget']},
                     ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'normal'
    if command == 'normal':
        normal()
    elif command == 'benchmark':
        benchmark(int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
    elif command == 'verify':
        verify()
    elif command == 'demo':
        demo()
    else:
        raise SystemExit('normal, benchmark [n], verify or demo')
