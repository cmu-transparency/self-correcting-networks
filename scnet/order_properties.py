import numpy as np
import tensorflow as tf


class SingleComparison(object):
    def __init__(self, x1, x2):
        self._x1 = x1
        self._x2 = x2

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    def __and__(self, other):
        if isinstance(other, SingleComparison):
            return ComparisonConjunction(self, other)

        elif isinstance(other, ComparisonConjunction):
            return ComparisonConjunction(self, *other.comparisons)

        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __or__(self, other):
        if isinstance(other, SingleComparison):
            return ComparisonDisjunction(self, other)

        elif isinstance(other, ComparisonConjunction):
            return ComparisonDisjunction(self, other)

        elif isinstance(other, ComparisonDisjunction):
            return ComparisonDisjunction(self, *other.comparisons)

        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __lt__(self, other):
        if isinstance(other, Output):
            return ComparisonConjunction(
                self,
                SingleComparison(self.x2, other.y))
        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __lshift__(self, other):
        return self < other

    def __gt__(self, other):
        if isinstance(other, Output):
            return ComparisonConjunction(
                self,
                SingleComparison(other.y, self.x2))
        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __rshift__(self, other):
        return self > other

    def __repr__(self):
        return f'y_{self.x1} < y_{self.x2}'

    def __str__(self):
        return repr(self)


class ComparisonConjunction(object):
    def __init__(self, *comparisons):
        self._comparisons = comparisons

    def __and__(self, other):
        if isinstance(other, ComparisonConjunction):
            return ComparisonConjunction(*self.comparisons, *other.comparisons)

        elif isinstance(other, SingleComparison):
            return ComparisonConjunction(*self.comparisons, other)

        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    @property
    def comparisons(self):
        return self._comparisons

    def __or__(self, other):
        if isinstance(other, SingleComparison):
            return ComparisonDisjunction(self, other)

        elif isinstance(other, ComparisonConjunction):
            return ComparisonDisjunction(self, other)

        elif isinstance(other, ComparisonDisjunction):
            return ComparisonDisjunction(self, *other.comparisons)

        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __repr__(self):
        return f'({" & ".join([str(comp) for comp in self.comparisons])})'

    def __str__(self):
        return repr(self)


class MaximalComparison(ComparisonConjunction):
    def __init__(self, c):
        self._c = c
        self._comparisons = [self]

    @property
    def c(self):
        return self._c

    def build(self, num_classes):
        return ComparisonConjunction(*[
            Output(self.c) > Output(j)
            for j in range(num_classes) if j != self.c
        ])

    def __repr__(self):
        return f'({self.c} is maximal)' 


class MinimalComparison(ComparisonConjunction):
    def __init__(self, c):
        self._c = c
        self._comparisons = [self]

    @property
    def c(self):
        return self._c

    def build(self, num_classes):
        return ComparisonConjunction(*[
            Output(self.c) < Output(j)
            for j in range(num_classes) if j != self.c
        ])

    def __repr__(self):
        return f'({self.c} is minimal)' 


class ComparisonDisjunction(object):
    def __init__(self, *comparisons):
        self._comparisons = [
            comparison if isinstance(comparison, ComparisonConjunction) else
            ComparisonConjunction(comparison)
            for comparison in comparisons
        ]

    @property
    def comparisons(self):
        return self._comparisons

    def __or__(self, other):
        if isinstance(other, ComparisonDisjunction):
            return ComparisonDisjunction(*self.comparisons, *other.comparisons)

        elif isinstance(other, ComparisonConjunction):
            return ComparisonDisjunction(*self.comparisons, other)

        elif isinstance(other, SingleComparison):
            return ComparisonDisjunction(*self.comparisons, other)

        else:
            raise ValueError(f'illegal argument type: {type(other)}')

    def __repr__(self):
        return f'({" | ".join([str(comp) for comp in self.comparisons])})'

    def __str__(self):
        return repr(self)


class OrderingPostcondition(object):
    def __init__(self, comparison):
        if isinstance(comparison, (SingleComparison, ComparisonConjunction)):
            self._disjunctions = ComparisonDisjunction(comparison)
        else:
            self._disjunctions = comparison

    @property
    def disjunctions(self):
        return self._disjunctions

    def make_graph(self, num_classes):
        disjuncts = []

        for disjunct in self._disjunctions.comparisons:
            graph = np.zeros((num_classes, num_classes), dtype='int32')

            if isinstance(disjunct, (MaximalComparison, MinimalComparison)):
                disjunct = disjunct.build(num_classes)

            for comparison in disjunct.comparisons:
                if isinstance(
                    comparison, (MaximalComparison, MinimalComparison)
                ):
                    for comparison in comparison.build(num_classes).comparisons:
                        graph[comparison.x2][comparison.x1] = 1

                else:
                    graph[comparison.x2][comparison.x1] = 1

            disjuncts.append(graph)

        return tf.constant(disjuncts)

    def __len__(self):
        return len(self._disjunctions.comparisons)

    def __getitem__(self, i):
        return self._disjunctions.comparisons[i]

    def __repr__(self):
        return repr(self._disjunctions)

    def __str__(self):
        return str(self._disjunctions)


class Output(object):
    def __init__(self, y):
        self._y = y

    @property
    def y(self):
        return self._y

    @property
    def is_maximal(self):
        return MaximalComparison(self._y)

    @property
    def is_minimal(self):
        return MinimalComparison(self._y)

    def __lt__(self, other):
        return SingleComparison(self.y, other.y)

    def __lshift__(self, other):
        return SingleComparison(self.y, other.y)

    def __gt__(self, other):
        return SingleComparison(other.y, self.y)

    def __rshift__(self, other):
        return SingleComparison(other.y, self.y)
