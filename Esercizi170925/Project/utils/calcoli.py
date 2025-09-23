def somma(a: int, b: int) -> int:
    """
    Somma due numeri interi.

    Parameters
    ----------
    a : int
        Primo numero intero.
    b : int
        Secondo numero intero.

    Returns
    -------
    int
        La somma dei due numeri.

    """
    return a + b

def conta_unici(lista):
    """
    Conta il numero di elementi unici in una lista.

    Parameters
    ----------
    lista : list
        La lista di elementi.

    Returns
    -------
    int
        Il numero di elementi unici nella lista.
    """
    return len(set(lista))

def primi_fino_a_n(n):
    """
    Restituisce una lista dei numeri primi fino a n.

    Parameters
    ----------
    n : int
        Il numero fino al quale cercare i numeri primi.

    Returns
    -------
    list
        Una lista dei numeri primi fino a n.

    Examples
    --------    
    >>> primi_fino_a_n(10)
    [2, 3, 5, 7]
    >>> primi_fino_a_n(20)
    [2, 3, 5, 7, 11, 13, 17, 19]
    """
    c = []
    for num in range(2, n + 1):
        primo = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                primo = False
                break
        if primo:
            c.append(num)
    return c