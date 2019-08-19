import naive_bayes as nb
import numpy as np

f_n = []
f_p = []
acc = []
print("f_n  |   f_p   |   acc")
for i in range(50):
    temp_f_n, temp_f_p, temp_acc = nb.train_and_test(nb.X, nb.y)
    f_n.append(temp_f_n)
    f_p.append(temp_f_p)
    acc.append(temp_acc)
    print(temp_f_n, " | ", temp_f_p, " | ", temp_acc)

f_n = np.array(f_n)
f_p = np.array(f_p)
acc = np.array(acc)
print("mean f_n = ", f_n.sum() / len(f_n))
print("mean f_p = ", f_p.sum() / len(f_p))
print("mean acc = ", acc.sum() / len(acc))