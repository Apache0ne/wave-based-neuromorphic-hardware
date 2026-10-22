"""Reproducible generated K-12 through precalculus reference questions.

Ground truth comes directly from independently constructed operands/formulas,
not from calling the solver under test.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import sympy as sp

@dataclass(frozen=True)
class Question:
    kind: str
    prompt: str
    expected: object

KINDS=("addition","subtraction","multiplication","division","fractions",
       "order_of_operations","percent","linear_equation","quadratic_equation",
       "expand","factor","rational_simplify","triangle_area","circle_area",
       "trigonometry","logarithm","radicals","linear_system","composition",
       "arithmetic_sequence","geometric_sequence")


def questions(seed: int, per_kind: int = 40) -> list[Question]:
    rng=random.Random(seed)
    result=[]
    r=lambda a,b:rng.randint(a,b)
    trig=[('sin(pi/6)',sp.Rational(1,2)),('sin(pi/3)',sp.sqrt(3)/2),
          ('cos(pi/3)',sp.Rational(1,2)),('cos(pi/4)',sp.sqrt(2)/2),
          ('tan(pi/4)',sp.Integer(1)),('sin(pi/2)',sp.Integer(1)),
          ('cos(pi)',sp.Integer(-1)),('tan(pi/6)',sp.sqrt(3)/3)]
    for kind in KINDS:
        for _ in range(per_kind):
            a,b,c,d=r(-90,90),r(-90,90),r(-9,9),r(-9,9)
            if kind=='addition': q=f'Evaluate: ({a}) + ({b})'; ans=a+b
            elif kind=='subtraction': q=f'Evaluate: ({a}) - ({b})'; ans=a-b
            elif kind=='multiplication': q=f'Evaluate: ({a}) * ({b})'; ans=a*b
            elif kind=='division':
                numerator=r(-30,30); denom=r(1,13)
                q=f'Evaluate: ({numerator}) / ({denom})'; ans=sp.Rational(numerator,denom)
            elif kind=='fractions':
                n1,n2=r(-9,9),r(-9,9); d1,d2=r(1,12),r(1,12)
                q=f'Evaluate: ({n1})/({d1}) + ({n2})/({d2})'; ans=sp.Rational(n1,d1)+sp.Rational(n2,d2)
            elif kind=='order_of_operations':
                v,w,z=r(-12,12),r(-12,12),r(-12,12)
                q=f'Evaluate: ({v}) + ({w}) * ({z})'; ans=v+w*z
            elif kind=='percent':
                p,n=r(1,100),r(1,500)
                q=f'Percent: {p}% of {n}'; ans=sp.Rational(p*n,100)
            elif kind=='linear_equation':
                coeff=r(1,12)*rng.choice((-1,1)); root=r(-20,20); offset=r(-35,35)
                q=f'Solve for x: {coeff}*x + ({offset}) = {coeff*root+offset}'; ans=root
            elif kind=='quadratic_equation':
                root1,root2=r(-12,12),r(-12,12)
                while root1==root2: root2=r(-12,12)
                q=f'Solve for x: x**2 + ({-root1-root2})*x + ({root1*root2}) = 0'
                ans=tuple(sorted((sp.Integer(root1),sp.Integer(root2)),key=sp.default_sort_key))
            elif kind=='expand':
                c1,c2=r(-15,15),r(-15,15)
                q=f'Expand: (x + ({c1}))*(x + ({c2}))'
                ans=sp.Symbol('x')**2+(c1+c2)*sp.Symbol('x')+c1*c2
            elif kind=='factor':
                c1,c2=r(-15,15),r(-15,15)
                q=f'Factor: x**2 + ({c1+c2})*x + ({c1*c2})'
                x=sp.Symbol('x'); ans=sp.factor((x+c1)*(x+c2))
            elif kind=='rational_simplify':
                c1,c2=r(-10,10),r(-10,10)
                while c1==c2:c2=r(-10,10)
                q=f'Simplify: ((x + ({c1}))*(x + ({c2})))/(x + ({c1}))'
                ans=sp.Symbol('x')+c2 # defined when x!=-c1, a domain caveat
            elif kind=='triangle_area':
                base,height=r(1,60),r(1,60)
                q=f'Triangle area: base={base}, height={height}'; ans=sp.Rational(base*height,2)
            elif kind=='circle_area':
                radius=r(1,30)
                q=f'Circle area: radius={radius}'; ans=sp.pi*radius**2
            elif kind=='trigonometry':
                s,ans=rng.choice(trig);q='Evaluate: '+s
            elif kind=='logarithm':
                base=r(2,8); power=r(0,5)
                q=f'Evaluate: log({base**power}, {base})'; ans=power
            elif kind=='radicals':
                n=r(1,20); coeff=r(1,12)
                q=f'Evaluate: sqrt({n*n*coeff})'; ans=sp.sqrt(n*n*coeff)
            elif kind=='linear_system':
                x0,y0=r(-12,12),r(-12,12)
                ca,cb,cc,cd=r(-6,6),r(-6,6),r(-6,6),r(-6,6)
                while ca*cd-cb*cc==0:
                    ca,cb,cc,cd=r(-6,6),r(-6,6),r(-6,6),r(-6,6)
                q=f'Solve system: {ca}*x+({cb})*y={ca*x0+cb*y0}; {cc}*x+({cd})*y={cc*x0+cd*y0}'
                ans=(sp.Integer(x0),sp.Integer(y0))
            elif kind=='composition':
                af,bf,ag,bg=r(-6,6),r(-12,12),r(-6,6),r(-12,12);v=r(-15,15)
                q=f'Compose: f(x)={af}*x+({bf}); g(x)={ag}*x+({bg}); evaluate f(g({v}))'
                ans=af*(ag*v+bg)+bf
            elif kind=='arithmetic_sequence':
                first,diff,n=r(-20,20),r(-10,10),r(1,30)
                q=f'Arithmetic sequence: first={first}, difference={diff}, n={n}'
                ans=first+(n-1)*diff
            elif kind=='geometric_sequence':
                first,ratio,n=r(-10,10),r(-3,3),r(1,9)
                q=f'Geometric sequence: first={first}, ratio={ratio}, n={n}'
                ans=first*(ratio**(n-1))
            else: raise AssertionError(kind)
            result.append(Question(kind,q,sp.Integer(ans) if isinstance(ans,int) else ans))
    rng.shuffle(result)
    return result
