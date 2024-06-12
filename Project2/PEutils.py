import numpy as np
from utils import kronString, stateToBinary, reverseOrder


""" 
###################################
Qubit ordering: left: highest, right lowest (completely stupid and annoying, but now it works more or less. I would suggest a different ordering!!)

Example: 01010: 0th qubit is the one at the right

If we want to apply a X gate to the first qubit, we need to apply the gate

I x I x I x X x I


This files contains all the functions I used for the phase estimation. In the jupyter notebook, "testingPE.ipynb" I have copied all of the functions
and added some print() commands to figure out, where things might go wrong and to check if my functions do what they are supposed to.


###################################
"""




H1 = 1/np.sqrt(2)*np.array([[1,1],[1,-1]], dtype=complex)
I1= np.array([[1,0],[0,1]],dtype=complex)
X1 = np.array([[0,1],[1,0]])
#projection operators
p0 =  np.array([[1,0],[0,0]],dtype=complex)
p1 = np.array([[0,0],[0,1]],dtype=complex)

def omega(i,j,n):
    return np.exp(2*np.pi*1j/2**n * i*j)

def QFTMatrix(n):
    # n: number of qubits
    N = 2**n
    # create a n x n dimensional np.array with ones in it
    operator = np.ones((N,N), dtype=complex)
    # now fill in the lower right square of the matrix with the powers of omega
    for i in range(1,N):
        for j in range(1,N):
            operator[i,j] = omega(i,j,n)
    # add normalization
    operator = operator*1/(N**0.5)
    return operator

def kronStringPE(string, operator):
    """
    Copy of kronString() from QFT code, changed a little bit for PE application
    Turn a string with single qubit gate instructions into kronecker products. We need to take into account, that the string
    needs to be iterated through backwards, because for example a X gate on the 1st qubit with 5 qubits in total would be
    I x Ix I x X x I.

    Input: string: String with single Qubit names, power: angle for phase gate

    Output: Kronecker product of all input gates, numpy array of shape 2**len(string)
    
    """
    strList = list(string)
    #print(strList)
    strList = strList[::-1]
    if strList[0] == "U":
        prod = operator
    elif strList[0] == "P":
        prod = p0
    elif strList[0] == "Q":
        prod = p1
    else:
        prod = I1
    for matrix in strList[1:]:
        if matrix == "U":
            prod = np.kron(operator,prod)
        elif matrix == "P":
            prod = np.kron(p0,prod)
        elif matrix == "Q":
            prod = np.kron(p1,prod)
        else:
            prod = np.kron(I1,prod)
    return prod

# copy of the CNOT function from the QFT, we can just swap the X with the unitary
def CU(controlling,t,n, operator):
    """
    A CNOT gate can be writte as CU,1=|0⟩⟨0|⊗I+|1⟩⟨1|⊗U for example.
    The outer products are at the position of the first index, whereas the X is on the right at the position of the second index.
    another example: CNOT1,3=I⊗|0⟩⟨0|⊗I⊗I+I⊗|1⟩⟨1|⊗I⊗X

    In the PE case, the controlling bit is some different one from the upper register and the controlled one is always the lower register.
    Therefore we just ignor U while creating the upper part for of the gate for the controlling part and then after the loop, add the controlled
    part. Here we take into account, that U might be bigger than 2x2 so we add a number of identities, that matches the qubits needed, for U.
    """
    # create strings for both parts to feed into kronString() function to calculate the products
    part1 = ""; part2 = ""
    """ 
    we need to reverse the controlling index passed from the applyU() function. This is because applyU() gives qubit index 0 for the
    most right qubit in the t register.
    """
    for i in range(t):
        if i == t-1-controlling:
            # p0 = P, p1 = Q, my kron function gets screwed up, because of list(str) for two letter names
            part1 += "P"
            part2 += "Q"
        else:
            part1 += "I" 
            part2 += "I" 
    for i in range(n):
        part1 += "I" 
        part2 += "U" 
    gate = kronStringPE(part1, operator) + kronStringPE(part2, operator)
    return gate


def applyHadamards(t,n, state):
    """ 
    Qubit ordering described at the top!

    Hadamard on the 0th qubit (with t=3) would be: I x I x H
    We can easily check if the Hadamard operation is correct, by looking at the binary representation of our state
    after the hadamard gates. The qubits representing the eigenstate, shouldnt have chagned, whereas all the other ones
    should exist as 0 and 1

    """
    # Apply Hadamard gate to all qubits of register t
    for i in range(t):
        # create a hadamard gate string for the qubits in t
        gateStr = ""
        for j in range(t):
            if j == i:
                gateStr = "H" + gateStr
            else:
                gateStr = "I" + gateStr
        # add identities for the lower register
        for i in range(n):
            gateStr = gateStr + "I" 
        #print(f"Shape of gate: {kronString(gateStr).shape[0]}, shape of state {state.shape[0]}")
        state = np.matmul(kronString(gateStr), state)

    return state

def applyInverseQFT(t,unitary):
    # first we need to add identites make the QFT matrix bigger and add identitys, so it  can be applied on the full
    # register but leaves the lower register unchanged
    
    finalQFT = np.kron(QFTMatrix(t).conj().T, np.identity(unitary.shape[0]))
    #print("QFT Matrix:")
    #print(np.round(finalQFT,2))
    return finalQFT

def applyU(t,n,state, unitary):
    # for each qubit in the upper register
    for i in range(t):
        # we start with the 0th qubit (most right qubit, befor the n eigenstate qubits)
        # apply U i times: for example. for the t-1th bit we apply U 2^(t-1) times
        controlledU = CU(controlling = i, t=t, n =n, operator = unitary)
        for j in range(2**(i)):
            # create the controlled phase gate that puts the phase from U onto the one qubit
            state = np.matmul(controlledU, state)
    return state

def calculatePhase(t,n, unitary, eigenstate):

    # prepare eigenstate in lower register
    upperState = np.zeros(2**t)
    upperState[0] = 1
    state = np.kron(upperState, eigenstate)
    state = applyHadamards(t,n,state)
    # Now apply exponentiated U gates to upper register
    state = applyU(t,n,state,unitary)
    # now to inverse QFT
    state = np.matmul(applyInverseQFT(t, unitary),state)

    """ 
    Now we need to extract the phases from the final state. This is a bit more complicated because of my choice of representation. I can choose
    between the general state in the 2**(t+n) product space, or I can choose the binary representation. Because you can't really read out of the 
    product space, which qubit is in which state, I will have to use the binary representation. The algorithm for this is following:

    1. Our final state is a super position of many states. So we need to iterate through every binary state.
    2. For each of the state, we calculate the total phase. For example first state: "0 00" where the space
    indicates, where computational and eigenstate register split (remember, that we reversed the qubits at some point, when you look at the output, you see
    that the first qubit (after the reversing order) is always 0, as it should be, because it's the unchanging eigenstate (in the case of a 2x2 Unitary matrix)).
    Here the phase gain is 0. For "0 10" we would have "Eigenstate Phi1 Phi0", so we would have a phase gain of 

    amplitude of this state * 2^-1
    """

    # convert state into binary
    state = reverseOrder(state,t+n)
    # tupel representation [("0101", amplitude)]
    binaryState = stateToBinary(state, t+n)
    binaryCombinations = [tup[0] for tup in binaryState]
    amplitudes =  [tup[1] for tup in binaryState]
    phase = 0
    # 1. iterate through all the states
    for combination in binaryCombinations:
        # transfer it into list of integers
        combinationint = [int(i) for i in combination]
        #print(combination)
        # the first n qubits are reserved for the eigenstate
        for i in range(n,t+n):
            # the most right qubit gets 1 as a factor, the second right gets 0.5 and so on
            # we are going from left to right, so we have start with the biggest fraction. First one is 2**(-(t-1))
            # add prints again to check
            if combinationint[i] == 1:
                #print(2**(-1*(t-1+n-i)))
                phase += amplitudes[binaryCombinations.index(combination)]*2**(-1*(t-i+1))

    return phase

def phaseEstimation(t,unitary):

    # calculate to be estimated values
    eigenvalues, eigenstates = np.linalg.eig(unitary)
    
    # convert eigenvalues to phases
    eigenvalues = np.log(eigenvalues) / (2*np.pi*1j)
    # make eigenvalues positiv
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < 0:
            eigenvalues[i] += 1
    #print(f"Eigenstate: {eigenstates[0]}")
    n = int(np.log2(unitary.shape[0]))
    phase = calculatePhase(t,n, unitary, eigenstates[0])
    #print(f"To be evaluated eigenvalue: {eigenvalues[0]}")
    #print(f"Estimated eigenvalue: {phase} with {t} qubits of precision.")
    return eigenvalues[0], phase

