import json
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


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == 'normal':
        normal()
    elif sys.argv[1] == 'benchmark':
        benchmark(int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
    else:
        raise SystemExit('normal or benchmark [n]')
