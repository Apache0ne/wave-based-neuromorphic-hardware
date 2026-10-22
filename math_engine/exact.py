"""Safe, exact reference math evaluator for reproducible K-12/precalculus tests.

This is a CPU/SymPy baseline and verifier, not a spin-wave implementation.
Only an allowlisted expression grammar is accepted (no eval/parse_expr).
"""
from __future__ import annotations
import ast
import re
from typing import Union
import sympy as sp

x, y = sp.symbols('x y')
SYMBOLS = {'x': x, 'y': y, 'pi': sp.pi, 'E': sp.E}
FUNCS = {'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
         'log': sp.log, 'exp': sp.exp, 'Abs': sp.Abs}
MAX_CHARS = 250
MAX_NODES = 70
MAX_EXP = 24

class MathInputError(ValueError):
    pass


def expr(source: str) -> sp.Expr:
    if len(source) > MAX_CHARS:
        raise MathInputError('expression too long')
    try:
        root = ast.parse(source.replace('^', '**'), mode='eval')
    except SyntaxError as exc:
        raise MathInputError('invalid expression syntax') from exc
    if sum(1 for _ in ast.walk(root)) > MAX_NODES:
        raise MathInputError('expression too complex')

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            if abs(node.value) > 1_000_000:
                raise MathInputError('integer too large')
            return sp.Integer(node.value)
        if isinstance(node, ast.Name) and node.id in SYMBOLS:
            return SYMBOLS[node.id]
        if isinstance(node, ast.UnaryOp):
            arg = visit(node.operand)
            if isinstance(node.op, ast.UAdd): return arg
            if isinstance(node.op, ast.USub): return -arg
        if isinstance(node, ast.BinOp):
            a = visit(node.left); b = visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div):
                if b == 0: raise MathInputError('division by zero')
                return a / b
            if isinstance(node.op, ast.Pow):
                if b.is_number is not True or not b.is_real or abs(b) > MAX_EXP:
                    raise MathInputError('unsupported exponent')
                return a ** b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
            args = [visit(a) for a in node.args]
            if not 1 <= len(args) <= 2:
                raise MathInputError('unsupported function arity')
            if len(args) == 2 and node.func.id != 'log':
                raise MathInputError('only log accepts two arguments')
            return FUNCS[node.func.id](*args)
        raise MathInputError(f'unsupported syntax: {type(node).__name__}')
    try:
        return visit(root)
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as exc:
        raise MathInputError(str(exc)) from exc


def solution_key(v):
    if isinstance(v, (list, tuple, sp.Tuple)):
        return tuple(str(sp.simplify(t)) for t in v)
    return str(sp.simplify(v))


def solve(prompt: str) -> Union[sp.Expr, tuple]:
    """Solve a restricted family of exact-worded questions without answer metadata."""
    if prompt.startswith('Evaluate: '):
        result = expr(prompt[10:])
        if result.free_symbols: raise MathInputError('evaluate needs constants only')
        return sp.simplify(result)
    if prompt.startswith('Simplify: '):
        return sp.cancel(expr(prompt[10:]))
    if prompt.startswith('Expand: '):
        return sp.expand(expr(prompt[8:]))
    if prompt.startswith('Factor: '):
        return sp.factor(expr(prompt[8:]))
    if prompt.startswith('Solve for x: '):
        eq = prompt[len('Solve for x: '):].split('=')
        if len(eq) != 2: raise MathInputError('expected one equation')
        a, b = (expr(t.strip()) for t in eq)
        if ((a.free_symbols | b.free_symbols) - {x}):
            raise MathInputError('expected x only')
        roots = sp.solve(sp.Eq(a, b), x)
        if not roots: raise MathInputError('no finite solutions')
        ordered = tuple(sorted(roots, key=sp.default_sort_key))
        return ordered[0] if len(ordered) == 1 else ordered
    if prompt.startswith('Percent: '):
        match = re.fullmatch(r'(-?\d+)% of (-?\d+)', prompt[9:])
        if not match: raise MathInputError('bad percent question')
        p, whole = map(int, match.groups())
        return sp.Rational(p * whole, 100)
    if prompt.startswith('Triangle area: '):
        match = re.fullmatch(r'base=(\d+), height=(\d+)',prompt[15:])
        if not match: raise MathInputError('bad triangle question')
        b, h = map(int, match.groups())
        return sp.Rational(b*h, 2)
    if prompt.startswith('Circle area: '):
        match = re.fullmatch(r'radius=(\d+)',prompt[13:])
        if not match: raise MathInputError('bad circle question')
        return sp.Integer(int(match.group(1)))**2 * sp.pi
    if prompt.startswith('Solve system: '):
        eqs = prompt[len('Solve system: '):].split(';')
        if len(eqs) != 2: raise MathInputError('expected two equations')
        parsed=[]
        for e in eqs:
            lhs, rhs=e.split('=')
            parsed.append(sp.Eq(expr(lhs.strip()),expr(rhs.strip())))
        solutions=sp.solve(parsed,(x,y),dict=True)
        if len(solutions)!=1: raise MathInputError('no unique solution')
        return (solutions[0][x],solutions[0][y])
    if prompt.startswith('Compose: '):
        match = re.fullmatch(r'f\(x\)=(.+); g\(x\)=(.+); evaluate f\(g\((-?\d+)\)\)',prompt[9:])
        if not match: raise MathInputError('bad function composition')
        f,g,n=match.groups()
        inner=expr(g.strip()).subs(x,sp.Integer(n))
        return sp.simplify(expr(f.strip()).subs(x,inner))
    if prompt.startswith('Arithmetic sequence: '):
        match=re.fullmatch(r'first=(-?\d+), difference=(-?\d+), n=(\d+)',prompt[21:])
        if not match: raise MathInputError('bad arithmetic sequence')
        first,d,n=map(int,match.groups())
        return sp.Integer(first+(n-1)*d)
    if prompt.startswith('Geometric sequence: '):
        match=re.fullmatch(r'first=(-?\d+), ratio=(-?\d+), n=(\d+)',prompt[20:])
        if not match: raise MathInputError('bad geometric sequence')
        first,r,n=map(int,match.groups())
        return sp.Integer(first*r**(n-1))
    raise MathInputError('unsupported question format')
