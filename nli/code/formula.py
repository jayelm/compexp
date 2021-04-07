import random

import pyparsing as pp


class F:
    def __init__(self):
        self.mask = None


class Leaf(F):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return str(self.val)

    def to_str(self, namer, sort=False):
        return namer(self.val)

    def __len__(self):
        return 1

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"Leaf({str(self)})"

    def get_vals(self):
        return [self.val]

    def is_leaf(self):
        return True


class Node(F):
    def is_leaf(self):
        return False


class UnaryNode(Node):
    arity = 1
    op = None

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return f"({self.op} {self.val})"

    def to_str(self, namer, sort=False):
        op_name = self.val.to_str(namer)
        return f"({self.op} {op_name})"

    def __len__(self):
        return 1 + len(self.val)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"{self.op}({self.val})"

    def get_vals(self):
        return self.val.get_vals()


class BinaryNode(Node):
    arity = 2
    op = None

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"

    def to_str(self, namer, sort=False):
        left_name = self.left.to_str(namer, sort=sort)
        right_name = self.right.to_str(namer, sort=sort)
        if not sort or (left_name < right_name):
            return f"({left_name} {self.op} {right_name})"
        else:
            return f"({right_name} {self.op} {left_name})"

    def __len__(self):
        return len(self.left) + len(self.right)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"{self.op}({self.left}, {self.right})"

    def get_vals(self):
        vals = []
        vals.extend(self.left.get_vals())
        vals.extend(self.right.get_vals())
        return vals


class Not(UnaryNode):
    op = "NOT"


class Neighbors(UnaryNode):
    op = "NEIGHBORS"


class Hypernym(UnaryNode):
    op = "HYPERNYM"


class Or(BinaryNode):
    op = "OR"


class And(BinaryNode):
    op = "AND"


UNARY_OPS = [Not, Neighbors, Hypernym]
BINARY_OPS = [Or, And]


# The most unnecessary thing I've ever done
identifier = pp.Word(pp.alphas.lower() + pp.nums + "-_ :.'\",;+-!?`$")(
    "FirstExpression"
)
condition = pp.Group(identifier)("MainBody")

# define AND, OR, and NOT as keywords, with "operator" results names
AND_ = pp.Keyword("AND")("operator")
OR_ = pp.Keyword("OR")("operator")
NOT_ = pp.Keyword("NOT")("operator")
NEIGHBORS_ = pp.Keyword("NEIGHBORS")("operator")
HYPERNYM_ = pp.Keyword("HYPERNYM")("operator")

expr = pp.operatorPrecedence(
    condition,
    [
        (
            NOT_,
            1,
            pp.opAssoc.RIGHT,
        ),
        (
            NEIGHBORS_,
            1,
            pp.opAssoc.RIGHT,
        ),
        (
            HYPERNYM_,
            1,
            pp.opAssoc.RIGHT,
        ),
        (
            AND_,
            2,
            pp.opAssoc.LEFT,
        ),
        (
            OR_,
            2,
            pp.opAssoc.LEFT,
        ),
    ],
)

# undocumented hack to assign a results name to (expr) - RED FLAG
expr.expr.resultsName = "group"


def parse(fstr, reverse_namer=lambda x: x):
    """
    Parse a string representation back into formula.
    Reverse_namer converts back from names to actual integer indices
    """
    flist = expr.parseString(fstr)[0]  # extract item 0 from single-item list
    return parse_flist(flist, reverse_namer)


def parse_flist(flist, reverse_namer):
    if len(flist) == 1:
        # Leaf
        val = flist[0].strip()
        return Leaf(reverse_namer(val))
    elif len(flist) == 2:
        # Unary op
        if flist[0] == "NOT":
            op = Not
        elif flist[0] == "NEIGHBORS":
            op = Neighbors
        else:
            raise ValueError(f"Unknown unary op {flist[0]}")
        val = parse_flist(flist[1], reverse_namer)
        return op(val)
    elif len(flist) == 3:
        # Binary op
        if flist[1] == "OR":
            op = Or
        elif flist[1] == "AND":
            op = And
        else:
            raise ValueError(f"Unknown binary op {flist[1]}")
        left = parse_flist(flist[0], reverse_namer)
        right = parse_flist(flist[2], reverse_namer)
        return op(left, right)
    else:
        raise ValueError(f"Could not parse {flist}")


def minor_negate(f, hard=False):
    """
    Negate a leaf

    If Hard, deterministically choose the one that's furthest away (TODO: Can
    be improved; neegate ORs earlier). Otherwise chooses randomly
    """
    if isinstance(f, Leaf):
        return Not(f)
    elif isinstance(f, Not):
        # Special case: if the val is a leaf, just return the val itself
        if isinstance(f.val, Leaf):
            return f.val
        else:
            return Not(minor_negate(f.val, hard=hard))
    elif isinstance(f, And):
        # Binary
        if hard:
            cond = len(f.left) < len(f.right)
        else:
            cond = random.random() < 0.5
        if cond:
            return And(f.left, minor_negate(f.right, hard=hard))
        else:
            return And(minor_negate(f.left, hard=hard), f.right)
    elif isinstance(f, Or):
        # Binary
        if hard:
            cond = len(f.left) < len(f.right)
        else:
            cond = random.random() < 0.5
        if cond:
            return Or(f.left, minor_negate(f.right, hard=hard))
        else:
            return Or(minor_negate(f.left, hard=hard), f.right)
    else:
        raise RuntimeError
