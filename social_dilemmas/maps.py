# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apple spawn point
# ' ' is empty space

HARVEST_MAP = [
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    "@ P   P      A    P AAAAA    P  A P  @",
    "@  P     A P AA    P    AAA    A  A  @",
    "@     A AAA  AAA    A    A AA AAAA   @",
    "@ A  AAA A    A  A AAA  A  A   A A   @",
    "@AAA  A A    A  AAA A  AAA        A P@",
    "@ A A  AAA  AAA  A A    A AA   AA AA @",
    "@  A A  AAA    A A  AAA    AAA  A    @",
    "@   AAA  A      AAA  A    AAAA       @",
    "@ P  A       A  A AAA    A  A      P @",
    "@A  AAA  A  A  AAA A    AAAA     P   @",
    "@    A A   AAA  A A      A AA   A  P @",
    "@     AAA   A A  AAA      AA   AAA P @",
    "@ A    A     AAA  A  P          A    @",
    "@       P     A         P  P P     P @",
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
]

CLEANUP_MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@RRRRRR     BBBBB@",
    "@HHHHHH      BBBB@",
    "@RRRRRR     BBBBB@",
    "@RRRRR  P    BBBB@",
    "@RRRRR    P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR      BBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@RRRRR   P P BBBB@",
    "@HHHHH   P  BBBBB@",
    "@RRRRRR    P BBBB@",
    "@HHHHHH P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH    P  BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH  P P BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH      BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]

# Where T is for the box trap (13 across x 11 down)
BOX_TRAPPED_MAP = [
    "@@@@@@@@@@@@@",
    "@           @",
    "@           @",
    "@  AA   TTT @",
    "@  AA   TPT @",
    "@       TTT @",
    "@  P     AA @",
    "@  AA    AA @",
    "@  AA       @",
    "@           @",
    "@@@@@@@@@@@@@"
]

# 'S' means turned-on switch
# 's' means turned-off switch
# 'D' means closed door
# 'd' means opened door
class SwitchMapElements:
    top_row = "@@@D@@@"
    empty_row = "@     @"
    one_switch_row = "@s    @"
    two_switch_row = "@s   s@"
    bottom_row = "@@@@@@@"


def view_map_size(map_):
    return len(map_[0]), len(map_)


if __name__ == '__main__':
    print("Harvest:", view_map_size(HARVEST_MAP))
    print("Cleanup:", view_map_size(CLEANUP_MAP))


