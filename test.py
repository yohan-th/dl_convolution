# *************************************************************************** #
#   ___  ____  __    ____  __   ____  __     __   ____    tf_training.py      #
#  / __)(  __)(  )  (  __)/ _\ (  _ \(  )   / _\ (  _ \                       #
# ( (_ \ ) _)  )(    ) _)/    \ ) _ (/ (_/\/    \ ) _ (   Created by yohan    #
#  \___/(__)  (__)  (__) \_/\_/(____/\____/\_/\_/(____/   19/06/2019 11:34:42 #
#                                                                             #
#           Contact: yohan.thollet@gfi.fr                 Updated by yohan    #
#                                                         19/06/2019 15:39:42 #
# *************************************************************************** #


def factorial(n):
  if n == 0:
      return 1
  else:
      return n * factorial(n - 1)