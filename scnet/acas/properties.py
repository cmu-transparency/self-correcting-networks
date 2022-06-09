import numpy as np

from scnet.properties import Property
from scnet.properties import InputInRange
from scnet.order_properties import OrderingPostcondition
from scnet.order_properties import Output


STRONG_LEFT = Output(3)
LEFT = Output(1)
COC = Output(0)
RIGHT = Output(2)
STRONG_RIGHT = Output(4)

MEANS = np.array([1.9791091e+04, 0., 0., 650., 600.])
RANGES = np.array([60261., 6.28318530718, 6.28318530718, 1100., 1200.])

norm = lambda x: (np.array(x) - MEANS) / RANGES

# phi_2: COC not minimal.
# ρ ≥ 55947.691, vown ≥ 1145, vint ≤ 60.
pre2 = InputInRange(
    norm([55947.691, -np.inf, -np.inf,   1145, -np.inf]),
    norm([   np.inf,  np.inf,  np.inf, np.inf,      60]))
post2 = OrderingPostcondition(
    COC >> STRONG_LEFT | COC >> LEFT | COC >> RIGHT | COC >> STRONG_RIGHT)

phi2 = Property(pre2, post2, 'φ2')

relevant_networks2 = [
    (i, j) 
    for i in range(5) for j in range(9)
    if i >= 1
]

# phi_3: COC not maximal.
# 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ ≥ 3.10, vown ≥ 980, vint ≥ 960.
pre3 = InputInRange(
    norm([1500, -0.06,   3.10,    980,    960]),
    norm([1800,  0.06, np.inf, np.inf, np.inf]))
post3 = OrderingPostcondition(
    COC << STRONG_LEFT | COC << LEFT | COC << RIGHT | COC << STRONG_RIGHT)

phi3 = Property(pre3, post3, 'φ3')

relevant_networks3 = [
    (i, j) 
    for i in range(5) for j in range(9)
    if (i != 0 or j != 6) and (i != 0 or j != 7) and (i !=0 or j != 8)
]

# phi_4: COC not maximal.
# 1500 ≤ ρ ≤ 1800, −0.06 ≤ θ ≤ 0.06, ψ = 0, vown ≥ 1000, 700 ≤ vint ≤ 800.
pre4 = InputInRange(
    norm([1500, -0.06, 0,   1000, 700]),
    norm([1800,  0.06, 0, np.inf, 800]))
post4 = OrderingPostcondition(
    COC << STRONG_LEFT | COC << LEFT | COC << RIGHT | COC << STRONG_RIGHT)

phi4 = Property(pre4, post4, 'φ4')

relevant_networks4 = [
    (i, j) 
    for i in range(5) for j in range(9)
    if (i != 0 or j != 6) and (i != 0 or j != 7) and (i !=0 or j != 8)
]

# phi_5: STRONG_RIGHT is maximal.
# 250 ≤ ρ ≤ 400, 0.2 ≤ θ ≤ 0.4, −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 
# 100 ≤ vown ≤ 400, 0 ≤ vint ≤ 400.
pre5 = InputInRange(
    norm([250, 0.2,         -np.pi, 100,   0]),
    norm([400, 0.4, -np.pi + 0.005, 400, 400]))
post5 = OrderingPostcondition(STRONG_RIGHT.is_maximal)

relevant_networks5 = [(0, 0)]

phi5 = Property(pre5, post5, 'φ5')

# phi_6: COC is maximal.
# 12000 ≤ ρ ≤ 62000, (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤ θ ≤ −0.7), 
# −3.141592 ≤ ψ ≤ −3.141592 + 0.005, 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200
pre6a = InputInRange(
    norm([12000,   0.7,         -np.pi,  100,    0]),
    norm([62000, np.pi, -np.pi + 0.005, 1200, 1200]))
pre6b = InputInRange(
    norm([12000, -np.pi,         -np.pi,  100,    0]),
    norm([62000,   -0.7, -np.pi + 0.005, 1200, 1200]))
post6 = OrderingPostcondition(COC.is_maximal)

phi6a = Property(pre6a, post6, 'φ6a')
phi6b = Property(pre6b, post6, 'φ6b')

relevant_networks6 = [(0, 0)]

# phi_7: STRONG_LEFT and STRONG_RIGHT are not maximal.
# 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ 3.141592, −3.141592 ≤ ψ ≤ 3.141592, 
# 100 ≤ vown ≤ 1200, 0 ≤ vint ≤ 1200.
pre7 = InputInRange(
    norm([    0, -np.pi, -np.pi,  100,    0]),
    norm([60760,  np.pi,  np.pi, 1200, 1200]))
post7a = OrderingPostcondition(
    STRONG_LEFT << LEFT | 
    STRONG_LEFT << COC | 
    STRONG_LEFT << RIGHT | 
    STRONG_LEFT << STRONG_RIGHT)
post7b = OrderingPostcondition(
    STRONG_RIGHT << STRONG_LEFT | 
    STRONG_RIGHT << LEFT | 
    STRONG_RIGHT << COC | 
    STRONG_RIGHT << RIGHT)

phi7a = Property(pre7, post7a, 'φ7a')
phi7b = Property(pre7, post7b, 'φ7b')

relevant_networks7 = [(0, 8)]

# phi_8: LEFT or COC is maximal.
# 0 ≤ ρ ≤ 60760, −3.141592 ≤ θ ≤ −0.75·3.141592, −0.1 ≤ ψ ≤ 0.1, 
# 600 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
pre8 = InputInRange(
    norm([    0,        -np.pi, -0.1,  600,  600]),
    norm([60760, -0.75 * np.pi,  0.1, 1200, 1200]))
post8 = OrderingPostcondition(LEFT.is_maximal | COC.is_maximal)

phi8 = Property(pre8, post8, 'φ8')

relevant_networks8 = [(1, 8)]

# phi_9: STRONG_LEFT is maximal.
# 2000 ≤ ρ ≤ 7000, −0.4 ≤ θ ≤ −0.14, −3.141592 ≤ ψ ≤ −3.141592 + 0.01, 
# 100 ≤ vown ≤ 150, 0 ≤ vint ≤ 150.
pre9 = InputInRange(
    norm([2000, -0.40,        -np.pi, 100,   0]),
    norm([7000, -0.14, -np.pi + 0.01, 150, 150]))
post9 = OrderingPostcondition(STRONG_LEFT.is_maximal)

phi9 = Property(pre9, post9, 'φ9')

relevant_networks9 = [(2, 2)]

# phi_10: COC is maximal.
# 36000 ≤ ρ ≤ 60760, 0.7 ≤ θ ≤ 3.141592, −3.141592 ≤ ψ ≤ −3.141592 + 0.01, 
# 900 ≤ vown ≤ 1200, 600 ≤ vint ≤ 1200.
pre10 = InputInRange(
    norm([36000,   0.7,        -np.pi,  900,  600]),
    norm([60760, np.pi, -np.pi + 0.01, 1200, 1200]))
post10 = OrderingPostcondition(COC.is_maximal)

phi10 = Property(pre10, post10, 'φ10')

relevant_networks10 = [(3, 4)]

# Empty property, in case any model has no properties required, since I'm not
# sure what will happen if we pass the repair layer an empty set of properties.
pre_empty = InputInRange(
    [ 1.,  1,  1,  1,  1],
    [-1., -1, -1, -1, -1])
post_empty = OrderingPostcondition(LEFT >> RIGHT & RIGHT >> LEFT)

phi_empty = Property(pre_empty, post_empty, 'EMPTY')


properties_for_model = {
    (i, j): [] for i in range(5) for j in range(9)
}
for model in relevant_networks2:
    properties_for_model[model].append(phi2)

for model in relevant_networks3:
    properties_for_model[model].append(phi3)

for model in relevant_networks4:
    properties_for_model[model].append(phi4)

for model in relevant_networks5:
    properties_for_model[model].append(phi5)

for model in relevant_networks6:
    properties_for_model[model].append(phi6a)
    properties_for_model[model].append(phi6b)

for model in relevant_networks7:
    properties_for_model[model].append(phi7a)
    properties_for_model[model].append(phi7b)

for model in relevant_networks8:
    properties_for_model[model].append(phi8)

for model in relevant_networks9:
    properties_for_model[model].append(phi9)

for model in relevant_networks10:
    properties_for_model[model].append(phi10)

for model in properties_for_model:
    if not properties_for_model[model]:
        properties_for_model[model].append(phi_empty)


safe_models = [
    model 
    for model in properties_for_model 
    if phi2 not in properties_for_model[model] and 
        phi8 not in properties_for_model[model]
]

unsafe_models = [
    model 
    for model in properties_for_model 
    if phi2 in properties_for_model[model] or 
        phi8 in properties_for_model[model]
]
