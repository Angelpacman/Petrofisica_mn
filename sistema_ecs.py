
"""
ahora toca armar el sistema de ecuaciones y resolverlo para todas las filas
"""
# define matrix A using Numpy arrays
A = np.matrix([ [189, 43.5, 55.5,   120],
                [1.0, 0.02, -0.035, 0.33],
                [1.0, 2.87, 2.65,   2.35],
                [1.0, 1.0,  1.0,    1.0]    ])
A.shape

#define matrix B
b = np.matrix([ [DT],
                [NPHI],
                [RHOB],
                [1] ])
# b = np.array([73.9477, 0.1275, 2.6503, 1])
B = np.array([59.8739, 0.0606, 2.5407, 1])

x = np.around(np.linalg.solve(A, B), decimals = 4)
x
#datos.head()

A_inverse = np.linalg.inv(A)

X = A_inverse * b
X

FIP  =  X[0]
VDOL =  X[1]
VSIL =  X[2]
VARC =  X[3]


#La idea de poner 2 shape es para que el arreglo quede de tamaño (400,1)
#hasta el momento no he encontrado como optimizar este detalle
FIP = np.array(FIP.T)[0]
FIP.shape
FIP = np.array(FIP.T)[0]
FIP.shape

VDOL = np.array(VDOL.T)[0]
VDOL.shape
VDOL = np.array(VDOL.T)[0]
VDOL.shape

VSIL = np.array(VSIL.T)[0]
VSIL.shape
VSIL = np.array(VSIL.T)[0]
VSIL.shape

VARC = np.array(VARC.T)[0]
VARC.shape
VARC = np.array(VARC.T)[0]
VARC.shape

VCAL = np.array(VARC*0.0000)

datos['VDOL'] = np.around(VDOL,decimals = 4)
datos['VCAL'] = np.around(VCAL,decimals = 4)
datos['VSIL'] = np.around(VSIL,decimals = 4)
datos['VARC'] = np.around(VARC,decimals = 4)
datos['FIP']  = np.around(FIP, decimals = 4)
datos.head()
