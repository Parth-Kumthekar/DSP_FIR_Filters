'''Create a Python FIR filter class which implements an FIR filter which has a method of the
form value dofilter(self,value) where both the value argument and return value are scalars
and not vectors (!) so that it can be used in a realtime system. The constructor of the class
takes the coefficients as its input:

class FIRfilter:
def __init__(self,_coefficients):
# your code here
def dofilter(self,v):
# your code here
return result'''