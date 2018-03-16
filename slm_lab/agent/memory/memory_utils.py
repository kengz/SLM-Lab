import numpy


class SumTree:
    ''' This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = numpy.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')


if __name__ == "__main__":
    tree = SumTree(10)
    print(f'Write idx: {tree.write}')
    tree.add(3, 0)
    tree.add(8, 1)
    tree.add(1, 2)
    tree.add(2, 3)
    tree.add(1, 4)
    tree.add(7, 5)
    print(f'Total: {tree.total()}')
    tree.print_tree()
    # print(f'S: {25} result: {tree.get(25)}')
    print(f'S: {1} result: {tree.get(1)}')
    print(f'S: {5} result: {tree.get(5)}')
    print(f'S: {10} result: {tree.get(10)}')
    print(f'S: {15} result: {tree.get(15)}')
    print(f'S: {20} result: {tree.get(20)}')
    print(f'S: {22} result: {tree.get(22)}')
