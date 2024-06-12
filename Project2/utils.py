"""
This file contains all the functions for the QFT algorithm and the direct calculation of the QFT with a matrix.


"""
import numpy as np

################# DFT as one matrix #########################################
def omega(i,j,n):
    return np.exp(2*np.pi*1j/2**n * i*j)

def QFTMatrix(state, n):
    """ 
    Just the big matrix with the powers of omega in it. The name might be confusing, but I used this name in different files so 
    it would take some trouble changing it.
    """

    # n: number of qubits
    N = 2**n
    # create a n x n dimensional np.array with ones in it
    operator = np.ones((N,N), dtype=complex)
    # now fill in the lower right square of the matrix with the powers of omega
    for i in range(1,N):
        for j in range(1,N):
            operator[i,j] = omega(i,j,n)
    # round for better view and add normalization
    operator = operator*1/(N**0.5)
    result = np.round(np.matmul(operator,state),4)
    return result, operator

# same as QFT but changed the sign in the omega-exponentiation, could also just use QFT(state,n).conj().T
def inverseQFT(state,n):
    # n: number of qubits
    N = n**2
    # create a n x n dimensional np.array with ones in it
    operator = np.ones((N,N), dtype=complex)
    # now fill in the lower right square of the matrix with the powers of omega
    for i in range(1,N):
        for j in range(1,N):
            operator[i,j] = omega(n)**(-i*j)
    # round for better view and add normalization
    operator = operator*1/(N**0.5),4
    result = np.round(np.matmul(operator,state),4)
    return result


############### Own QFT algorithm #############################################
import numpy as np

H1 = 1/np.sqrt(2)*np.array([[1,1],[1,-1]], dtype=complex)
I1= np.array([[1,0],[0,1]],dtype=complex)
X1 = np.array([[0,1],[1,0]])
#projection operators
p0 =  np.array([[1,0],[0,0]],dtype=complex)
p1 = np.array([[0,0],[0,1]],dtype=complex)
""" we could theoretically only have the 1 qubit matrices and construct the multi qubit matrices in the QFT
function as well, however I think dividing it makes more sense, which is why I have functions that do actions like
"Apply hadamard to first qubit" or something similar"""


# single qubit phase gate
# we have pi/2 the first time, pi/4 the scond and so on
def phase(index):
    print(f"{-1/2**index}pi")
    return np.array([[1,0],[0,np.exp(np.pi * 1j* 1/2**index)]])


def binaryToState(binaryList, n):
    """
    Convert a list of binary state representations into a list of basis state vectors.
    
    Parameters:
    binaryList: list containing tuples of (binary state representation, coefficient)
    n: number qubits
    
    Returns:
    np.array: normalized linear combination of all states that appear in the binary representation
    """
    state = np.zeros(2**n, dtype=complex)
    for binaryStr, coefficient in binaryList:
        # check dimensions
        if len(binaryStr) != n:
            raise ValueError(f"Binary string '{binaryStr}' does not match the number of qubits: {n}")
        
        # turn binary into decimal
        index = int(binaryStr, 2)
        
        # create basis vector
        basisVector = np.zeros(2**n)
        basisVector[index] = 1
        
        # Add the basis vector to the list
        state += coefficient * basisVector
    
    return state

def stateToBinary(state,n):
    """
    Convert a basis state vector into a list of binary state representations.
    
    Parameters:
    state (np.array): A vector representing the basis state in the computational basis
    
    Returns:
    list of str: A list of strings representing the binary states with their corresponding coefficients
    """
    # list of tuples (binary State representation, coefficient)
    binaryList = []
    
    for i, coefficient in enumerate(state):
        if coefficient != 0:
            # Convert the index to a binary string
            binaryStr = format(i, '0' + str(n) + 'b')
            # add the left zeroes, if they were forgotten to add
            while len(binaryStr) != n:
               binaryStr = "0" + binaryStr 
            binaryList.append((binaryStr, coefficient))
    return binaryList


# turn a String like XIXY into kronecker products to make code look a bit better (only for I and X now, because we need Swap gates)
def kronString(string, power=0):
    """
    Turn a string with single qubit gate instructions into kronecker products. We need to take into account, that the string
    needs to be iterated through backwards, because for example a X gate on the 1st qubit with 5 qubits in total would be
    I x I x I x X x I.

    Input: string: String with single Qubit names, power: angle for phase gate

    Output: Kronecker product of all input gates, numpy array of shape 2**len(string)
    
    """
    strList = list(string)
    #print(strList)
    strList = strList[::-1]
    #print(f"Lenght of stirng list: {len(strList)}")
    if strList[0] == "X":
        prod = X1
    elif strList[0] == "P":
        prod = p0
    elif strList[0] == "H":
        prod = H1
    elif strList[0] == "Q":
        prod = p1
    elif strList[0] == "S":
        prod = phase(power)
    else:
        prod = I1
    for matrix in strList[1:]:
        if matrix == "X":
            prod = np.kron(X1,prod)
        elif matrix == "P":
            prod = np.kron(p0,prod)
        elif matrix == "Q":
            prod = np.kron(p1,prod)
        elif matrix == "H":
            prod = np.kron(H1,prod)
        elif matrix == "S":
            prod = np.kron(phase(power),prod)
        else:
            prod = np.kron(I1,prod)
    return prod

def CNOT(controlling, controlled,n, printString=False):
    """
    A CNOT gate can be writte as CNOT0,1=|0⟩⟨0|⊗I+|1⟩⟨1|⊗X for example.
    The outer products are at the position of the first index, whereas the X is on the right at the position of the second index.
    another example: CNOT1,3=I⊗|0⟩⟨0|⊗I⊗I+I⊗|1⟩⟨1|⊗I⊗X
    """
    # create strings for both parts to feed into kronString() function to calculate the products
    part1 = ""; part2 = ""
    for i in range(n):
        if i == controlling:
            # p0 = P, p1 = Q, my kron function gets screwed up, because of list(str) for two letter names
            part1 += "P"
            part2 += "Q"
        elif i == controlled:
            part1 += "I"
            part2 += "X"
        else:
            part1 += "I" 
            part2 += "I" 
    if printString == True:
        print(part1,part2)
    gate = kronString(part1) + kronString(part2)
    return gate

def swap(lower,higher,n):
    """
    SWAP gates can be decomposed into 3 CNOT gates. For example CNOT0,1 CNOT1,0 CNOT0,1 to swap the qubits 0 and 1
    """
    gate = np.identity(2**n, dtype=complex)
    # first CNOT
    gate = np.matmul(CNOT(lower, higher,n),gate)
    # second CNOT, with reversed indices
    gate = np.matmul(CNOT(higher,lower,n), gate)
    # third CNOT, back to ordering of first case
    gate = np.matmul(CNOT(lower,higher,n),gate)
    return gate

def reverseOrder(state,n):
    """
    We need to reverse the order of the qubits to make the circuit fit the theoretical description. For that we will apply swap gates from
    the inner qubit pair all the way to the outer qubit pair
    """
    # middle start index, for example 5 qubits, we start with 1 (for the second qubit): i = 1, j = 3
    # for example 8 qubits: i = 3 (4th qubit), j = 4
    i = n//2 - 1
    # distinguish even from odd qubit number
    if n%2 != 0:
        j = i+2
    else:
        j = i+1
    while i >= 0:
        #print(f"Now swapping qubit {i} with {j}")
        state = np.matmul(swap(i,j,n),state)
        i-=1
        j+=1
    return state


# controlled phase gate, phases are powers of omega. For example for a 3 qubit case we apply p/2 and pi/4 to the first qubit
# here we follow chapter 2.7s formula on how to express controlled gates with Identity and Projection operators
def Cphase(power,indexC, indexA,n):
    # indexA: index where phase is applied to, indexC: controlling index, n: number of qubits, power: power for angle (pi/2 or pi/4 ...)
    #phase = np.array([[0,1],[1,0]])
    # define both right ends of the products for both parts of formula 2.1 in hundt
    part1 = ""; part2 = "" 
    for i in range(n):
        if i == indexC:
            part1 += "P" 
            part2 += "Q"
        elif i == indexA:
            part1 += "I"
            # S string for phase gate
            part2 += "S"
        else:
            part1 += "I"
            part2 += "I" 
    gate = kronString(part1,power) + kronString(part2,power)
    return gate

def H(a, n):
    """
    Creating a Hadamard gate to be applied to one qubit a in a n qubit system

    Input:  int a - position where to apply hadamard
            int n - number of qubits
    
    Output: np.array of shape 2**n x 2**n - hadamard gate
    
    """
    # a is the ath qubit we want to apply the hadamard gate to and n is the number of qubits
    if a >= n:
        print("You are a trying to apply to a qubit that doesn't exist...")
    # according to Robert hundt, if we have two qubits and want to apply H to the first we get H x I
    # so we need to reverse the order
    reversed = n-1-a
    if reversed == 0:
        firstfactor = H1
    else:
        firstfactor = I1
    result = firstfactor
    # do consecutive tensor products with H1 or I1 depending on which position H1 is supposed to be applied
    for i in range(1,n):
        if reversed == i:
            factor = H1
        else:
            factor = I1
        result = np.kron(factor, result)
    return result

def QFT(state,n):
    """
    Calculating the DFT of a state via the QFT algorithm
    
    Input:  np.array of dimension 2**n - state to be transformed
            int n - number of qubits

    Output: np.array of dimension 2**2 - transfomred state
    
    """

    # while we are not at the last qubit
    i = 0
    operator = np.identity(2**n)
    while i < n:
        # apply hadamard to the ith qubit
        print(f"Apply hadamard to {i}th qubit")
        #print(H(i,n))
        operator = np.matmul(H(i,n), operator)
        state = np.matmul(H(i,n), state)
        k = i
        # now apply phase to all qubits from 0 to n-1 (excluding)
        # (for example first round in a 3 qubit qft we apply 2 phase gates: k= 0, 1; 1 = n-2)
        power = 1 # first phase gate gets pi/2, next then pi/4 ... we always start with pi/2
        while k <= n-2:
            print(f"Apply CPhase with power {power}, IndexC {k+1}, indexA {i}")
            #operator = np.matmul(Cphase(power, k+1,i, n), operator)
            state = np.matmul(Cphase(power, k+1,i, n), state)
            #print("Cphse")
            #print(Cphase(power, k+1,i,n))
            power += 1
            k+=1
        i+=1
    # Now after putting the circuit into code, we need to swap the order of the qubits to match the definition of the QFT
    state = reverseOrder(state, n)
    operator = 1
    return np.round(state,4), operator








