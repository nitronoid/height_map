import math

def normalize(N):
  norm = math.sqrt(N[0]*N[0] + N[1] * N[1])
  N[0] /= norm
  N[1] /= norm

def relq(n1, n2):
  # normalize
  normalize(n1)
  normalize(n2)
  print("normalized:", n1, n2)

  n = [0.0, 0.0]
  n[0] = n1[0] - n2[0];
  n[1] = n1[1] - n2[1];
  print("n1 - n2: {}".format(n))

  q = abs(n[1] / n[0] if n[0] else 0)
  grad1 = n1[1] / n1[0] if n1[0] else float('inf')
  grad2 = n2[1] / n2[0] if n2[0] else float('inf')
  print(grad1, grad2)
  if grad1 > grad2:
    q = -q
  return q


n2 = [-1.0, 0.1000001]
n1 = [-1.0, 0.1]

print("result", relq(n1, n2))

