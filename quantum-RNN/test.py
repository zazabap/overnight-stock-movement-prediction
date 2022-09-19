# Author: Shiwen An 
# Date: 2022.08.04 
# Purpose: Try to make some Testing Neural Network and its Circuit. 
import time
import matplotlib
import numpy as np
import paddle

from numpy import pi as PI
from matplotlib import pyplot as plt

from paddle import matmul, transpose, reshape
from paddle_quantum.ansatz import Circuit
from paddle_quantum.gate import BasisEncoding, AmplitudeEncoding, AngleEncoding, IQPEncoding
from paddle_quantum.gate import IQPEncoding

from paddle_quantum.qinfo import pauli_str_to_matrix # N qubits Pauli matrix
from paddle_quantum.linalg import dagger  # complex conjugate
import paddle_quantum

import sklearn
from sklearn import svm
from sklearn.datasets import fetch_openml, make_moons, make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

from IPython.display import clear_output
from tqdm import tqdm

# import tutorial


def data():
  # class label, lepton 1 pT, lepton 1 eta, lepton 1 phi, lepton 2 pT, 
  # lepton 2 eta, lepton 2 phi, missing energy magnitude, missing energy phi, 
  # MET_rel, axial MET, M_R, M_TR_2, R, MT2, S_R, M_Delta_R, dPhi_r_b, cos(theta_r1)
  df = pd.read_csv("../data/HIGGS_100.csv",names=('isSignal',
      'lep1_pt','lep1_eta','lep1_phi','miss_ene' ,'miss_ene_phi', 
      'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
      'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag',
      'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag',
      'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag',
      'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb' ))

  feature_dim = 3  # dimension of each data point
  # the feature could also be divided into 3 5 7
  if feature_dim == 3:
      SelectedFeatures = ['isSignal', 'lep1_pt', 'jet_1_pt', 'miss_ene']
  elif feature_dim == 5:
      SelectedFeatures = ['isSignal', 'jet_1_pt','jet_2_pt','miss_ene','jet_3_pt','jet_4_pt']
  elif feature_dim == 7:
      SelectedFeatures = ['isSignal', 'lep1_pt','lep1_eta','lep2_pt','lep2_eta','miss_ene','M_TR_2','M_Delta_R']

  #print(df)
  #jobn = JOBN
  training_size = 80
  testing_size = 20
  #shots = 1024
  #uin_depth = NDEPTH_UIN
  #uvar_depth = NDEPTH_UVAR
  #niter = NITER
  #backend_name = 'BACKENDNAME'
  #option = 'OPTION'
  #random_seed = 10598+1010*uin_depth+101*uvar_depth+jobn

  print("Original Dataframe:")
  print(df)

  X = df[['lep1_pt', 'jet_1_pt', 'miss_ene']] 
  y = df[['isSignal']]

  print(X)
  print(y)

  # Train Test split
  df_sig =  df[SelectedFeatures]
  df_bkg =  df[SelectedFeatures]

#   print(df_sig['lep1_pt'])
#   print(df_bkg['lep1_pt'])

  df_sig = df_sig[df['isSignal'] == 1.0]
  df_bkg = df_bkg[df['isSignal'] == 0.0]


  print(df_sig)
  print(df_bkg)

  # Good place to start implementing the classification 
  # Algorithm
  Ntrain = 800
  Ntest = 200 
  Nqubit = 3 
  Depth = 1 
  Batch = 80 
  Epoch = int(200 * Batch/Ntrain)
  LR = 0.01 

  X = X.to_numpy()
  y = y.to_numpy()

  # Train Test Split for the set 
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

  print("X_train result")
  print(X_train)
  print("X_test result")
  print(X_test)
  print("y_train result")
  print(y_train)
  print("y_test result")
  print(y_test)

  datapoints_transform_to_state(X, 3)



  return df_sig, df_bkg

# Good Data point implementation
def datapoints_transform_to_state(data, n_qubits):
    """
    :param data: shape [-1, 2]
    :param n_qubits: the number of qubits to which
    the data transformed
    :return: shape [-1, 1, 2 ^ n_qubits]
        the first parameter -1 in this shape means can be arbitrary. In this tutorial, it equals to BATCH.
    """
    print(data)
    dim1, dim2 = data.shape
    print("The dimension for the data: ")
    print(dim1)

    res = []
    for sam in range(dim1):
        res_state = 1.
        zero_state = np.array([[1, 0, 0]])
        # Angle Encoding
        for i in range(n_qubits):
            # For even number qubits, perform Rz(arccos(x0^2)) Ry(arcsin(x0))
            if i  == 0:
                state_tmp=np.dot(zero_state, Ry(np.arcsin(data[sam][0])).T)
                state_tmp=np.dot(state_tmp, Rz(np.arccos(data[sam][0] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
            # For odd number qubits, perform Rz(arccos(x1^2)) Ry(arcsin(x1))
            elif i == 1:
                state_tmp=np.dot(zero_state, Ry(np.arcsin(data[sam][1])).T)
                state_tmp=np.dot(state_tmp, Rz(np.arccos(data[sam][1] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
            elif i == 2:
                state_tmp=np.dot(zero_state, Ry(np.arcsin(data[sam][2])).T)
                state_tmp=np.dot(state_tmp, Rz(np.arccos(data[sam][2] ** 2)).T)
                res_state=np.kron(res_state, state_tmp)
        res.append(res_state)
    res = np.array(res, dtype=paddle_quantum.get_dtype())

    print(res)
    
    return res


# Nice gate to look at
def Ry(theta):
    """
    :param theta: parameter
    :return: Y rotation matrix
    """
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

# Another nice gate to look at
def Rz(theta):
    """
    :param theta: parameter
    :return: Z rotation matrix
    """
    return np.array([[np.cos(theta / 2) - np.sin(theta / 2) * 1j, 0],
                     [0, np.cos(theta / 2) + np.sin(theta / 2) * 1j]])



def amplitude_encoding(df_sig):
  n = 2
  # Initialize the circuit
  built_in_amplitude_enc = AmplitudeEncoding(num_qubits=n)
  # Classical information x should be of type Tensor
  x = paddle.to_tensor([0.5, 0.5, 0.5])
  state = built_in_amplitude_enc(x)
  print(state)
    
def angle_encoding(df_sig):
  # Number of qubits = length of the classical information
  n = 3
  # Initialize the circuit
  angle_enc = Circuit(n)
  # X is the classical information
  x = paddle.to_tensor([np.pi, np.pi, np.pi], 'float64')
  # Add a layer of rotation y gates
  for i in range(len(x)):
    angle_enc.ry(qubits_idx=i, param=x[i])      
  print(angle_enc)

# In case our data has 3 dimensions
# 3 qubits for encoding. 
# Then prepare a group of initial 
# Quantum State |000>
# Encode the classical information into
# a group of quantum gates U(x) and 
# act them on the initial quantum states
# then |phi> = U(x)|000>

# One could also give m qubits to encode a two-dimensional 
# classical data point. 

# Robust Encoding
# https://arxiv.org/pdf/2003.01695.pdf


#   df_sig_training = df_sig.values[:training_size]
#   df_bkg_training = df_bkg.values[:training_size]
#   df_sig_test = df_sig.values[training_size:training_size+testing_size]
#   df_bkg_test = df_bkg.values[training_size:training_size+testing_size]
#   training_input = {'1':df_sig_training, '0':df_bkg_training}
#   test_input = {'1':df_sig_test, '0':df_bkg_test}


# Start Creating Circuit and work on the function
def create_circuit(num_qubits, depth):
    # step 1.1: Create an N qubit circuit
    circuit = paddle_quantum.ansatz.Circuit(num_qubits)
    # step 1.2: Add gates to each layer
    for _ in range(0, depth):
        circuit.rx('full')
        circuit.rz('full')
        circuit.ry('full')
        circuit.cnot('linear')
    return circuit

def example():
  psi_target = np.kron(
    np.kron(np.array([1, 0]), np.array([0, 1])),
    np.array([1/np.sqrt(2), 1/np.sqrt(2)])
  )  # <01+|
  psi_target = paddle_quantum.state.to_state(paddle.to_tensor(psi_target), dtype=paddle_quantum.get_dtype())
  fid_func = paddle_quantum.loss.StateFidelity(psi_target)


if __name__ == '__main__' :
  df1, df2 = data()
  amplitude_encoding(df1)
  angle_encoding(df1)
  num_qubits = 3
  depth = 1
  # cir = create_circuit(num_qubits, depth)
  # print(cir)