import tensorflow as tf

from abc import ABC as AbstractBaseClass
from abc import abstractmethod

from scnet.order_properties import OrderingPostcondition
from scnet.utils import rotate_row


class Property(object):
    def __init__(self, pre, post, name=None):
        self._pre = pre

        if isinstance(post, Disjunction):
            self._post = post.postconditions

        elif isinstance(post, OrderingPostcondition):
            self._post = post

        else:
            self._post = [post]

        self._name = name

    @property
    def pre(self):
        return self._pre

    @property
    def post(self):
        return self._post

    def __str__(self):
        if self._name:
            return f'{self._name}:\n{self.pre} ==>\n  {self.post}'
        
        return f'{self.pre} ==>\n  {self.post}'

    def __repr__(self):
        return str(self)


class Precondition(AbstractBaseClass):

    @abstractmethod
    def sat(self, x, y=None):
        raise NotImplementedError


class InputInRange(Precondition):

    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def sat(self, x, y=None):
        return tf.logical_and(
            tf.reduce_all(self.lower <= x, axis=1),
            tf.reduce_all(x <= self.upper, axis=1))


class SumSmallerThan(Precondition):
  
  def __init__(self, bound):
    self._bound = bound

  @property
  def bound(self):
    return self._bound

  def sat(self, x, y=None):
    return tf.reduce_sum(x, axis=1) <= self.bound


class SumGreaterThan(Precondition):
  
  def __init__(self, bound):
    self._bound = bound

  @property
  def bound(self):
    return self._bound

  def sat(self, x, y=None):
    return tf.reduce_sum(x,axis=1) >= self.bound


class ClassifiedAs(Precondition):
    def __init__(self, c):
        if not isinstance(c, (tuple, list)):
            c = [c]

        self._c = tf.constant([c])

    @property
    def c(self):
        return self._c.numpy().tolist()[0]

    def sat(self, x, y=None):
        return tf.reduce_any(
            tf.equal(tf.cast(tf.argmax(y, axis=1)[:,None], 'int32'), self.c), 
            axis=1)

    def __str__(self):
        return f'F(x) is in {self.c}'

    def __repr__(self):
        return str(self)


class Postcondition(AbstractBaseClass):
    
    @abstractmethod
    def check(self, y):
        raise NotImplementedError


class Disjunction(Postcondition):
    def __init__(self, *postconditions):
        # Make sure that the list of postconditions is always flat.
        self._postconditions = []
        for post in postconditions:
            if isinstance(post, Disjunction):
                for post_i in post:
                    self._postconditions.append(post_i)
            else:
                self._postconditions.append(post)

    @property
    def postconditions(self):
        return self._postconditions

    def check(self, y):
        return tf.reduce_any([
            postcondition.check(y) for postcondition in self.postconditions
        ])


class MinMaxOrdering(Postcondition):

    NOT_MAXIMAL = 0
    NOT_MINIMAL = 1
    MAXIMAL = 2
    MINIMAL = 3

    @staticmethod
    def from_class_requirement(classes, num_classes, requirement):
        if not isinstance(classes, (list, tuple)):
            classes = [classes]

        class_mask = tf.reduce_sum(
            tf.one_hot(classes, depth=num_classes), axis=0)

        mask = tf.clip_by_value(
            tf.one_hot(requirement, depth=4)[:,None] * class_mask[None], 
            0., 
            1.)[None]

        return MinMaxOrdering(mask)

    def __init__(self, mask):
        self._mask = mask

    @property
    def mask(self):
        return self._mask

    @staticmethod
    def conjunction(*properties):
        return MinMaxOrdering(
            tf.clip_by_value(
                tf.reduce_sum([p.mask for p in properties], axis=0), 0., 1.))

    @staticmethod
    def satisfiable(masks):
        return tf.reduce_all(
            [
                # Only one class can be maximal.
                tf.reduce_sum(masks[:,MinMaxOrdering.MAXIMAL], axis=1) <= 1,
                
                # Only one class can be minimal.
                tf.reduce_sum(masks[:,MinMaxOrdering.MINIMAL], axis=1) <= 1,
                
                # A class not be both maximal and not maximal.
                tf.reduce_all(
                    masks[:,MinMaxOrdering.MAXIMAL] + 
                        masks[:,MinMaxOrdering.NOT_MAXIMAL] <= 1,
                    axis=1),
                
                # A class not be both minimal and not minimal.
                tf.reduce_all(
                    masks[:,MinMaxOrdering.MINIMAL] + 
                        masks[:,MinMaxOrdering.NOT_MINIMAL] <= 1,
                    axis=1),
                
                # A class not be both maximal and minimal.
                tf.reduce_all(
                    masks[:,MinMaxOrdering.MAXIMAL] + 
                        masks[:,MinMaxOrdering.MINIMAL] <= 1,
                    axis=1),
                
                # There must exist two distinct classes that can be maximal and
                # minimal respectively.
                tf.reduce_prod(
                    masks[:,MinMaxOrdering.NOT_MAXIMAL] + 
                        masks[:,MinMaxOrdering.MINIMAL], 
                    axis=1) == 0,
                tf.reduce_prod(
                    masks[:,MinMaxOrdering.NOT_MINIMAL] + 
                        masks[:,MinMaxOrdering.MAXIMAL], 
                    axis=1) == 0,
                tf.reduce_sum(
                    tf.cast(
                        masks[:,MinMaxOrdering.NOT_MAXIMAL] * 
                            masks[:,MinMaxOrdering.NOT_MINIMAL] == 0,
                        'int32'),
                    axis=1) >= 2,
            ],
            axis=0)

    def fix(self, y):
        y = Maximal.do_fix(self.mask, y)
        y = Minimal.do_fix(self.mask, y)
        y = NotMaximal.do_fix(self.mask, y)
        y = NotMinimal.do_fix(self.mask, y)

        return y

    @staticmethod
    def do_fix(mask, y):
        raise NotImplementedError

    def check(self, y):
        return tf.reduce_all(
            [
                Maximal.do_check(self.mask, y),
                Minimal.do_check(self.mask, y),
                NotMaximal.do_check(self.mask, y),
                NotMinimal.do_check(self.mask, y),
            ], 
            axis=0)

    @staticmethod
    def do_check(mask, y):
        raise NotImplementedError

    def __str__(self):
        return str(self.mask)

    def __repr__(self):
        return repr(self.mask)

class NotMaximal(MinMaxOrdering):
    def __init__(self, disallowed_classes, num_classes):
        super().__init__(MinMaxOrdering.from_class_requirement(
            disallowed_classes, num_classes, MinMaxOrdering.NOT_MAXIMAL).mask)

    @staticmethod
    def do_fix(mask, y):
        mask = mask[:,MinMaxOrdering.NOT_MAXIMAL]

        mask_ordered_by_y = tf.gather(
            mask, tf.argsort(y, axis=1, direction='DESCENDING'), batch_dims=1)

        y_rolls = tf.concat(
            [
                tf.cast(
                    # The argmin selects the first class that is not precluded
                    # from being maximal, i.e., the first 1.
                    tf.argmin(mask_ordered_by_y, axis=1)[:,None], 
                    'float32'),
                y
            ],
            axis=1)

        return tf.map_fn(rotate_row('from_top'), y_rolls)

    @staticmethod
    def do_check(mask, y):
        mask = mask[:,MinMaxOrdering.NOT_MAXIMAL]

        num_classes = y.shape[1]

        return tf.reduce_all(
            tf.equal(tf.one_hot(tf.argmax(y, axis=1), num_classes) * mask, 0.),
            axis=1)

class NotMinimal(MinMaxOrdering):
    def __init__(self, disallowed_classes, num_classes):
        super().__init__(MinMaxOrdering.from_class_requirement(
            disallowed_classes, num_classes, MinMaxOrdering.NOT_MINIMAL).mask)

    @staticmethod
    def do_fix(mask, y):
        mask = mask[:,MinMaxOrdering.NOT_MINIMAL]

        mask_ordered_by_y = tf.gather(
            mask, tf.argsort(y, axis=1, direction='ASCENDING'), batch_dims=1)

        y_rolls = tf.concat(
            [
                tf.cast(
                    # The argmin selects the first class that is not precluded
                    # from being minimal, i.e., the first 1.
                    tf.argmin(mask_ordered_by_y, axis=1)[:,None], 
                    'float32'),
                y
            ],
            axis=1)

        return tf.map_fn(rotate_row('from_bottom'), y_rolls)

    @staticmethod
    def do_check(mask, y):
        mask = mask[:,MinMaxOrdering.NOT_MINIMAL]

        num_classes = y.shape[1]

        return tf.reduce_all(
            tf.equal(tf.one_hot(tf.argmin(y, axis=1), num_classes) * mask, 0.),
            axis=1)

class Maximal(MinMaxOrdering):
    def __init__(self, disallowed_classes, num_classes):
        super().__init__(MinMaxOrdering.from_class_requirement(
            disallowed_classes, num_classes, MinMaxOrdering.MAXIMAL).mask)

    @staticmethod
    def do_fix(mask, y):
        mask = mask[:,MinMaxOrdering.MAXIMAL]

        mask_ordered_by_y = tf.gather(
            mask, tf.argsort(y, axis=1, direction='DESCENDING'), batch_dims=1)

        y_rolls = tf.concat(
            [
                tf.cast(
                    # The argmax selects the first class that is required to be
                    # maximal, i.e., the first 0.
                    tf.argmax(mask_ordered_by_y, axis=1)[:,None], 
                    'float32'),
                y
            ],
            axis=1)

        return tf.map_fn(rotate_row('from_top'), y_rolls)

    @staticmethod
    def do_check(mask, y):
        mask = mask[:,MinMaxOrdering.MAXIMAL]

        num_classes = y.shape[1]

        return tf.logical_or(
            tf.reduce_all(tf.equal(mask, 0.), axis=1),
            tf.reduce_all(
                tf.equal(
                    tf.one_hot(tf.argmax(y, axis=1), num_classes), 
                    mask),
                axis=1))

class Minimal(MinMaxOrdering):
    def __init__(self, disallowed_classes, num_classes):
        super().__init__(MinMaxOrdering.from_class_requirement(
            disallowed_classes, num_classes, MinMaxOrdering.MINIMAL).mask)

    @staticmethod
    def do_fix(mask, y):
        mask = mask[:,MinMaxOrdering.MINIMAL]

        mask_ordered_by_y = tf.gather(
            mask, tf.argsort(y, axis=1, direction='ASCENDING'), batch_dims=1)

        y_rolls = tf.concat(
            [
                tf.cast(
                    # The argmax selects the first class that is required to be
                    # minimal, i.e., the first 0.
                    tf.argmax(mask_ordered_by_y, axis=1)[:,None], 
                    'float32'),
                y
            ],
            axis=1)

        return tf.map_fn(rotate_row('from_top'), y_rolls)

    @staticmethod
    def do_check(mask, y):
        mask = mask[:,MinMaxOrdering.MINIMAL]

        num_classes = y.shape[1]

        return tf.logical_or(
            tf.reduce_all(tf.equal(mask, 0.), axis=1),
            tf.reduce_all(
                tf.equal(
                    tf.one_hot(tf.argmin(y, axis=1), num_classes), 
                    mask),
                axis=1))
