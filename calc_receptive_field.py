def convrj(rin,jin):
    k = 3
    s = 1
    p = 1
    return calc_rj(rin, jin, k, s, p)


def poolrj(rin,jin):
    k = 4
    s = 2
    p = 2
    return calc_rj(rin, jin, k, s, p)
    return rout, jout


def calc_rj(rin, jin, k, s, p):
    jout = jin*s
    rout = rin + (k-1)*jin
    return (rout, jout)

    
def calc_recep():
    r = 1
    j = 1
    (r, j) = convrj(r, j)
    (r, j) = convrj(r, j)
    (r, j) = poolrj(r, j)
    
    (r, j) = convrj(r, j)
    (r, j) = convrj(r, j)
    (r, j) = poolrj(r, j)
    
    (r, j) = convrj(r, j)
    (r, j) = convrj(r, j)
    (r, j) = poolrj(r, j)
    
    (r, j) = convrj(r, j)
    (r, j) = convrj(r, j)
    (r, j) = poolrj(r, j)
    
    (r, j) = convrj(r, j)
    (r, j) = convrj(r, j)
    return (r, j) 
