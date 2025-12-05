def is_abelian(group_type, group_rank):
    if group_type == "GAUGE_U": 
        if group_rank == "rank_1": return True
        else: return False
    elif group_type == "GAUGE_SU": return False
    else: 
        print(f"Invalid group type: {group_type}")
        return None

def SU_N_dim(N, rep):
    """
    Dimension of the SU(N) representation.
    """
    if rep == "singlet": return 1
    elif rep in ("fnd", "anti_fnd"): return N
    elif rep == "adj": return N**2 - 1
    else: return None



def SU_N_Dynkin_label(N, rep):
    """
    Dynkin label of the SU(N) representation.
    """
    if rep == "singlet": return [0] * (N - 1)
    elif rep == "fnd": return [1] + [0] * (N - 2)
    elif rep == "anti_fnd": return [0] * (N - 2) + [1]
    elif rep == "adj":
        adjoint = [0] * (N - 1)
        adjoint[0] += 1
        adjoint[-1] += 1
        return adjoint
    else: return None



def SU_N_Dynkin_index(N, rep):
    """
    Dynkin index of the SU(N) representation.
    """
    label = SU_N_Dynkin_label(N, rep)
    dimR = SU_N_dim(N, rep)

    G = [[0.0] * (N - 1) for _ in range(N - 1)]
    for i in range(1, N):
        for j in range(1, N):
            G[i - 1][j - 1] = min(i, j) - (i * j) / N

    a_vec = label.copy()
    v = [x + 2.0 for x in a_vec]
    w = [sum(G[i][j] * v[j] for j in range(N - 1)) for i in range(N - 1)]
    inner = sum(a_vec[i] * w[i] for i in range(N - 1))
    C2 = 0.5 * inner
    dimG = N**2 - 1
    T = (dimR * C2) / dimG

    return float(dimR), float(C2), float(T)



def SU_N_cubic_anomaly(N, rep):
    """
    Cubic anomaly of the SU(N) representation.
    """
    if rep == "singlet": return 0
    elif rep == "fnd": return 1 if N >= 3 else 0    
    elif rep == "anti_fnd": return -1 if N >= 3 else 0
    elif rep == "adj": return 0
    else: return None


if __name__ == "__main__":
    test = SU_N_cubic_anomaly(3, "anti_fnd")
    print(test)
    