from scnet.properties import Property
from scnet.properties import ClassifiedAs
from scnet.order_properties import ComparisonConjunction
from scnet.order_properties import OrderingPostcondition
from scnet.order_properties import Output


superclasses = [
    [4, 30, 55, 72, 95],
    [1, 32, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 28, 61],
    [0, 51, 53, 57, 83],
    [22, 39, 40, 86, 87],
    [5, 20, 25, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 37, 68, 76],
    [23, 33, 49, 60, 71],
    [15, 19, 21, 31, 38],
    [34, 63, 64, 66, 75],
    [26, 45, 77, 79, 99],
    [2, 11, 35, 46, 98],
    [27, 29, 44, 78, 93],
    [36, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

def get_superclass_properties():
    return [
        Property(
            ClassifiedAs(superclass),
            OrderingPostcondition(ComparisonConjunction(*[
                Output(c_in) > Output(c_out)
                for c_in in superclass
                    for c_out in range(100)
                if c_out not in superclass
            ])))
        for superclass in superclasses
    ]
