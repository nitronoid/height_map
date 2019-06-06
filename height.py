import math

def normalize(N):
  norm = math.sqrt(N[0]*N[0] + N[1] * N[1])
  N[0] /= norm
  N[1] /= norm

def relq(n1, n2):
  # normalize
  normalize(n1)
  normalize(n2)
  print("normalized:", ["{:.15f}".format(i) for i in n1], ["{:.15f}".format(i) for i in n2])

  n = [0.0, 0.0]
  n[0] = n1[0] - n2[0];
  n[1] = n1[1] - n2[1];
  print("n1 - n2: {}".format(["{:.15f}".format(i) for i in n]))

  grad1 = abs(n1[1] / n1[0]) if n1[0] else float('inf')
  grad2 = abs(n2[1] / n2[0]) if n2[0] else float('inf')
  print(grad1, grad2)

  if n[0] :
    print("standard")
    q = abs(n[1] / n[0])
  else:
    print("x == 0")
    if grad1 == float('inf'):
      print("g1 == inf")
      q = 0.0
    elif grad1 == 0.0:
      print("g1 == 0")
      q = float('inf')
    else:
      print("sim")
      1.0/grad1

  if grad1 > grad2:
    q = -q
  return q


n1 = [0.00000000000853, 0.985]
n2 = [0.00000000000844, 0.986]

print("result", relq(n1, n2))

