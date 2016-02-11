
import numpy as np

from factorix.hermitian import hermitian_dot

if __name__ == '__main__':
    np.set_printoptions(precision=2)

    j = np.complex(0, -1)
    n_obj = 4
    emb_obj = [np.exp(j*np.pi*i/n_obj*2) for i in range(n_obj)]
    e0 = emb_obj[0]
    e1 = emb_obj[1]
    e2 = emb_obj[2]
    e3 = emb_obj[3]
    u = np.array([[e0], [e2], [e1], [e3+e2], [e0+e1]], dtype=complex)
    mat = (np.real(u.dot(np.conj(u.T))) + np.imag(u.dot(np.conj(u.T))))
    print(mat.round(3))
    # print(1 + j)
    #
    # mat = np.zeros((n_obj, n_obj))
    # for i in range(n_obj):
    #     mat[i, (i + 1) % n_obj] = 1
    #
    # print(np.round(emb_obj, 2))
    # print(np.round(mat, 2))

    ev, u0 = np.linalg.eig(mat)
    print(ev, u0)
    u = u0 * np.diag(np.sqrt(ev))
    print(u)
    sel = np.real(ev) ** 2 + np.imag(ev) ** 2 > 1e-3
    print(sel)
    u = u[:, sel]
    print(u)
    print(1, u0.dot(np.conj(u0.T)))
    print(2, u.dot(np.conj(u.T)))


