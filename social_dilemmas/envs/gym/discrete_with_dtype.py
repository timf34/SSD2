import numpy as np
from gym.spaces import Discrete

# TODO: this could probably be moved to a utils file.

class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype, start: int = 0):
        assert n >= 0
        self.n = n
        self.start = start
        # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
        super(Discrete, self).__init__((), dtype)


def test_discrete_with_d_type():
    x = DiscreteWithDType(8, dtype=np.uint8)

    print("vars(x):", vars(x))
    assert 'start' in vars(x)


if __name__ == "__main__":
    test_discrete_with_d_type()


