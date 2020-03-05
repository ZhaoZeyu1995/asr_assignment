def compute_alignment_errors(a, b):
    """
    
    a = reference, b = hypothesis
    
    Given two lists or strings, align and compute substitution, deletion and insertion counts
    
    Returns (num_subs, num_del, num_ins)
    """
    
    if type(a)==str:
        a = a.split()
    if type(b)==str:
        b = b.split()
    
    M = len(a)
    N = len(b)

    if M == 0:
        return (0, 0, N)

    if N == 0:
        return (0, M, 0)
    
    e = [[0]*(N+1) for i in range(M+1)]

    for n in range(N+1):
         e[0][n] = n

    for m in range(1,M+1):

        e[m][0] = e[m-1][0] + 1

        for n in range(1, N+1):

            sub_or_ok  = e[m-1][n-1] 

            if a[m-1] != b[n-1]:
                sub_or_ok += 1

            del_ = e[m-1][n]+ 1
            ins_ = e[m][n-1] + 1

            e[m][n] = min(sub_or_ok, ins_, del_)


    m = M
    n = N

    alignment = []  # not used in this version

    nsub, ndel, nins = (0,0,0)
    
    while m!=0 or n!=0:

        if m==0:
            last_m = m
            last_n = n-1
            nins+=1
        elif n==0:
            last_m = m-1
            last_n = n
            ndel+=1
        else:

            if a[m-1] != b[n-1]:
                sub_ = e[m-1][n-1] + 1
                ok_ = float('Inf')
            else:
                sub_ = float('Inf')
                ok_ = e[m-1][n-1]

            del_ = e[m-1][n] + 1
            ins_ = e[m][n-1] + 1

            # change to <= is prefer subs to ins/del
            if ok_ <= min(del_, ins_):           
                last_m = m-1
                last_n = n-1
            elif sub_ < min(del_, ins_):
                nsub+=1
                last_m = m-1
                last_n = n-1
            elif del_ < ins_:
                last_m = m-1
                last_n = n
                ndel+=1
            else:
                last_m = m
                last_n = n-1
                nins+=1

        if last_m == m:
            a_sym = '*'
        else:
            a_sym = a[last_m]

        if last_n == n:
            b_sym = '*'
        else:
            b_sym = b[last_n]

        # output.append((a_sym, b_sym))
        m = last_m
        n = last_n

    return (nsub, ndel, nins)
