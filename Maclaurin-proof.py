import math

def taylor_expansion_exp(x,num_terms):
  """
  approximate e^x using a taylor series expansion

  because d/dx(e^x) = e^x and e^0 = 1, the n-th derivative evaluated at 0 is always 1. Therefore, the n-th term of the series simplifies to (x^n) / n!
  """
  approximation  = 0.0
  print(f"--- Approximating e^{x} using {num_terms} terms ---")

  for n in range(num_terms):
    # property 1 & 2: the n-th derivative evaluated at 0 is always e^0 =1.
    nth_derivative_at_zero = 1.0
    # calculate x^n
    x_power = x**n
    # calculate n!
    factorial_n = math.factorial(n)
    # combine to form the taylor term : (f^(n)(0) * x^n) / n!
    term = (nth_derivative_at_zero * x_power ) / factorial_n
    approximation += term
    print(f"n = {n:<2} | Term: {term:<10.5f} | Current Sum: {approximation:.5f}")
  print("-" * 45)
  print(f"Final Approximation: {approximation:.7f}")
  print(f"Exact math.exp({x}):   {math.exp(x):.7f}")
    
  return approximation

#example: calculate e^2 using 10 terms

target_x = 2
terms_to_compute = 10

#taylor_expansion_exp(target_x,terms_to_compute )
#--- Approximating e^2 using 10 terms ---
#n = 0  | Term: 1.00000    | Current Sum: 1.00000
#n = 1  | Term: 2.00000    | Current Sum: 3.00000
#n = 2  | Term: 2.00000    | Current Sum: 5.00000
#n = 3  | Term: 1.33333    | Current Sum: 6.33333
#n = 4  | Term: 0.66667    | Current Sum: 7.00000
#n = 5  | Term: 0.26667    | Current Sum: 7.26667
#n = 6  | Term: 0.08889    | Current Sum: 7.35556
#n = 7  | Term: 0.02540    | Current Sum: 7.38095
#n = 8  | Term: 0.00635    | Current Sum: 7.38730
#n = 9  | Term: 0.00141    | Current Sum: 7.38871
#---------------------------------------------
#Final Approximation: 7.3887125
#Exact math.exp(2):   7.3890561
    
   
