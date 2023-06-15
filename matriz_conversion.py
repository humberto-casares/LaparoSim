import numpy as np
from sympy import symbols, lambdify

def conversion_1p(x1,y1,x2,y2):
    px1,px2,py1,py2 = symbols('px1 px2 py1 py2')
    X = 1.0e+16*(2.83018291311153e+31*px1*py2 + 2.73075661131684e+33*px1 + 1.15727573370151e+30*py1*py2 + 8.89929214310054e+32*py1 - 
    7.71917713573048e+33*py2 + 4.01662516700103e+35)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 
    2.55812598862112e+48*py1 + 2.87788996997844e+48*py2 + 3.37291219238725e+51) 
    Y = 400000000000000.0*(5.25498140596957e+31*px1*py2 + 4.69064340595306e+33*px1 - 6.10568343659077e+32*py1*py2 - 3.76029304645889e+34*py1 + 
    1.60512803395662e+35*py2 + 4.75609995875492e+37)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 
    2.55812598862112e+48*py1 + 2.87788996997844e+48*py2 + 3.37291219238725e+51)
    Z = (-1.22079871321883e+46*px1*py2 - 1.77082224241074e+48*px1 - 1.24689628116839e+46*py1*py2 + 2.9965406827307e+49*py1 - 9.58898996476242e+49*py2 + 
    3.68173087261855e+52)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 2.55812598862112e+48*py1 + 
    2.87788996997844e+48*py2 + 3.37291219238725e+51)

 
    X = lambdify([px1, py1, px2, py2], X)
    Y = lambdify([px1, py1, px2, py2], Y)
    Z = lambdify([px1, py1, px2, py2], Z)


    Xr = X(x1, y1, x2, y2)
    Yr = Y(x1, y1, x2, y2)
    Zr = Z(x1, y1, x2, y2)


    return np.round(Xr,6), np.round(Yr,6), np.round(Zr,6)

def conversion_2p(x1,y1,x2,y2,x3,y3,x4,y4):
    px1,px2,py1,py2 = symbols('px1 px2 py1 py2')
    X = 1.0e+16*(2.83018291311153e+31*px1*py2 + 2.73075661131684e+33*px1 + 1.15727573370151e+30*py1*py2 + 8.89929214310054e+32*py1 - 
    7.71917713573048e+33*py2 + 4.01662516700103e+35)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 
    2.55812598862112e+48*py1 + 2.87788996997844e+48*py2 + 3.37291219238725e+51) 
    Y = 400000000000000.0*(5.25498140596957e+31*px1*py2 + 4.69064340595306e+33*px1 - 6.10568343659077e+32*py1*py2 - 3.76029304645889e+34*py1 + 
    1.60512803395662e+35*py2 + 4.75609995875492e+37)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 
    2.55812598862112e+48*py1 + 2.87788996997844e+48*py2 + 3.37291219238725e+51)
    Z = (-1.22079871321883e+46*px1*py2 - 1.77082224241074e+48*px1 - 1.24689628116839e+46*py1*py2 + 2.9965406827307e+49*py1 - 9.58898996476242e+49*py2 + 
    3.68173087261855e+52)/(1.56867373577979e+45*px1*py2 + 1.12817446957939e+47*px1 + 4.81274128114974e+45*py1*py2 + 2.55812598862112e+48*py1 + 
    2.87788996997844e+48*py2 + 3.37291219238725e+51)

 
    X = lambdify([px1, py1, px2, py2], X)
    Y = lambdify([px1, py1, px2, py2], Y)
    Z = lambdify([px1, py1, px2, py2], Z)


    Xr = X(x1, y1, x2, y2)
    Yr = Y(x1, y1, x2, y2)
    Zr = Z(x1, y1, x2, y2)

    Xr1 = X(x3, y3, x4, y4)
    Yr1 = Y(x3, y3, x4, y4)
    Zr1 = Z(x3, y3, x4, y4)

    return np.round(Xr,6), np.round(Yr,6), np.round(Zr,6), np.round(Xr1,6), np.round(Yr1,6), np.round(Zr1,6)
