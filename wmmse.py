from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import copy
from copy import deepcopy
# import pickle as pkl
from pdb import set_trace as bp

def zero_forcing(channel_realization, total_power):
  
  ZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization)))))
  ZF_solution = ZF_solution*np.sqrt(total_power)/np.linalg.norm(ZF_solution) # scaled according to the power constraint

  return np.transpose(ZF_solution)


# Computes the regularized zero-forcing solution as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
def regularized_zero_forcing(channel_realization, total_power, regularization_parameter = 0, path_loss_option = False):
  
  if path_loss_option == False:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + nr_of_users/total_power*np.eye(nr_of_users, nr_of_users)))
  else:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + regularization_parameter*np.eye(nr_of_users, nr_of_users)))

  RZF_solution = RZF_solution*np.sqrt(total_power)/np.linalg.norm(RZF_solution) # scaled according to the power constraint

  return np.transpose(RZF_solution)

def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
  nr_of_BS_antennas = Phi_diag_elements.size
  mu_array = mu*np.ones(Phi_diag_elements.size)
  result = np.divide(Phi_diag_elements,(Sigma_diag_elements + mu_array)**2+1e-8)
  result = np.sum(result)
  return result

def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel,0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_id,:])))**2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id and user_index in selected_users:
        inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_index,:])))**2
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result

def compute_norm_of_complex_array(x):
  result = np.sqrt(np.sum((np.absolute(x))**2))
  return result

def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
   result = 0
   nr_of_users = np.size(channel,0)
   
   for user_index in range(nr_of_users):
     if user_index in selected_users:
       user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
       result = result + user_weights[user_index]*np.log2(1 + user_sinr)
    
   return result

def compute_min_rate(user_weights, channel, precoder, noise_power, selected_users):
   nr_of_users = np.size(channel,0)

   rates = []
   for user_index in range(nr_of_users):
     if user_index in selected_users:
       user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
       rates.append(np.log2(1 + user_sinr))
    
   return np.min(rates)

def run_WMMSE(epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations, log = False, init=None, power_tolerance=1e-4):

  nr_of_users = np.size(channel,0)
  nr_of_BS_antennas = np.size(channel,1)
  WSR=[] # to check if the WSR (our cost function) increases at each iteration of the WMMSE
  MR=[]
  break_condition = epsilon + 1 # break condition to stop the WMMSE iterations and exit the while
  receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
  mse_weights = np.ones(nr_of_users) # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
  transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))# transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)
  
  new_receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # for the first iteration 
  new_mse_weights = np.zeros(nr_of_users) # for the first iteration
  new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas)) # for the first iteration

  
  # Initialization of transmitter precoder
  if init is None:
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder[user_index,:] = channel[user_index,:]
    transmitter_precoder = transmitter_precoder/np.linalg.norm(transmitter_precoder)*np.sqrt(total_power)
  else:
    transmitter_precoder = init
    
  # Store the WSR obtained with the initialized trasmitter precoder    
  WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))

  # Compute the initial power of the transmitter precoder
  initial_power = 0
  for user_index in range(nr_of_users):
    if user_index in selected_users:
      initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index,:]))**2 
  if log == True:
    print("Power of the initialized transmitter precoder:", initial_power)

  nr_of_iteration_counter = 0 # to keep track of the number of iteration of the WMMSE

  while break_condition >= epsilon and nr_of_iteration_counter<=max_nr_of_iterations:
    
    nr_of_iteration_counter = nr_of_iteration_counter + 1
    if log == True:
      print("WMMSE ITERATION: ", nr_of_iteration_counter)

    # Optimize receiver precoder - eq. (5) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:
        user_interference = 0.0
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2

        new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_1,:]) / (noise_power + user_interference)

    # Optimize mse_weights - eq. (13) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:

        user_interference = 0 # it includes the channel of all selected users
        inter_user_interference = 0 # it includes the channel of all selected users apart from the current one
        
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        for user_index_2 in range(nr_of_users):
          if user_index_2 != user_index_1 and user_index_2 in selected_users:
            inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        
        new_mse_weights[user_index_1] = (noise_power + user_interference)/(noise_power + inter_user_interference)

    A = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))+1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        A = A + (new_mse_weights[user_index]*user_weights[user_index]*(np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Sigma_diag_elements_true, U = np.linalg.eigh(A)
    Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
    Lambda = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas)) + 1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    
    for user_index in range(nr_of_users):
      if user_index in selected_users:     
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        Lambda = Lambda + ((user_weights[user_index])**2)*((new_mse_weights[user_index])**2)*((np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Phi = np.matmul(np.matmul(np.conj(np.transpose(U)),Lambda),U)
    Phi_diag_elements_true = np.diag(Phi)
    Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
    Phi_diag_elements = np.real(Phi_diag_elements)

    for i in range(len(Phi_diag_elements)):
      if Phi_diag_elements[i]<np.finfo(float).eps:
        Phi_diag_elements[i] = np.finfo(float).eps
      if (Sigma_diag_elements[i])<np.finfo(float).eps:
        Sigma_diag_elements[i] = 0

    # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)
    power = 0 # the power of transmitter precoder (i.e. sum of the squared norm)
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        if np.linalg.det(A) != 0:
          w = np.matmul(np.linalg.inv(A),np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index])
          power = power + (compute_norm_of_complex_array(w))**2

    # If mu = 0 is a solution, then mu_star = 0
    if np.linalg.det(A) != 0 and power <= total_power:
      mu_star = 0
    # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
    else:
      power_distance = [] # list to store the distance from total_power in the bisection algorithm 
      mu_low = np.sqrt(1/total_power*np.sum(Phi_diag_elements))
      mu_high = 0
      low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
      high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

      obtained_power = total_power + 2*power_tolerance # initialization of the obtained power such that we enter the while 

      # Bisection search
      it = 0
      while np.absolute(total_power - obtained_power) > power_tolerance and it < 10000:
        it += 1
        mu_new = (mu_high + mu_low)/2
        obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new) # eq. (18) in the paper of Shi et al.
        power_distance.append(np.absolute(total_power - obtained_power))
        if obtained_power > total_power:
          mu_high = mu_new
        if obtained_power < total_power:
          mu_low = mu_new
      mu_star = mu_new
    #   if log == True:
    #     print("first value:", power_distance[0])
    #     plt.title("Distance from the target value in bisection (it should decrease)")
    #     plt.plot(power_distance)
    #     plt.show()

    for user_index in range(nr_of_users):
      if user_index in selected_users:
        new_transmitter_precoder[user_index,:] = np.matmul(np.linalg.inv(A + mu_star*np.eye(nr_of_BS_antennas)),channel[user_index,:])*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index]) 

    # To select only the weights of the selected users to check the break condition
    mse_weights_selected_users = []
    new_mse_weights_selected_users = []
    for user_index in range(nr_of_users): 
      if user_index in selected_users:
        mse_weights_selected_users.append(mse_weights[user_index])
        new_mse_weights_selected_users.append(new_mse_weights[user_index])

    mse_weights = deepcopy(new_mse_weights)
    transmitter_precoder = deepcopy(new_transmitter_precoder)
    receiver_precoder = deepcopy(new_receiver_precoder)

    WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))
    MR.append(compute_min_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))
    break_condition = np.absolute(np.log2(np.prod(new_mse_weights_selected_users))-np.log2(np.prod(mse_weights_selected_users)))

  if log == True:
    plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
    plt.plot(WSR,'bo')
    plt.show()

#   print(nr_of_iteration_counter)
#   print(np.linalg.norm(transmitter_precoder))

#   print('wmmse iterations:',nr_of_iteration_counter)
#   plt.plot(WSR)
  return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1], MR[-1]

# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option = False, path_loss_range = [-5,5] ):
  channel_nn = []
  initial_transmitter_precoder = []
  channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))

  
  for i in range(nr_of_users):

      regularization_parameter_for_RZF_solution = 0
      path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
      if path_loss_option == True:
        path_loss = np.random.uniform(path_loss_range[0],path_loss_range[-1])
        regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution + 1/((10**(path_loss/10))*total_power) # computed as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc

      result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
      result_imag  =  np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
      channel_WMMSE[i,:] = np.reshape(result_real,(1,nr_of_BS_antennas)) + 1j*np.reshape(result_imag, (1,nr_of_BS_antennas))
      result_col_1 = np.vstack((result_real,result_imag))
      result_col_2 = np.vstack((-result_imag,result_real))
      result =  np.hstack((result_col_1, result_col_2))
      initial_transmitter_precoder.append(result_col_1)
      channel_nn.append(result)

  initial_transmitter_precoder_array = np.array(initial_transmitter_precoder)
  initial_transmitter_precoder_array = np.sqrt(total_power)*initial_transmitter_precoder_array/np.linalg.norm(initial_transmitter_precoder_array)
  initial_transmitter_precoder = []

  for i in range(nr_of_users):
    initial_transmitter_precoder.append(initial_transmitter_precoder_array[i])

  return channel_nn, initial_transmitter_precoder, channel_WMMSE, regularization_parameter_for_RZF_solution

if __name__ == '__main__':
    # Set variables 
    nr_of_users = 16
    nr_of_BS_antennas = 16
    scheduled_users = np.arange(nr_of_users) # array of scheduled users. Note that we schedule all the users.
    epsilon = 1e-4 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
    power_tolerance = 1e-4 # used to end the bisection search in the WMMSE algorithm in Shi et al.
    total_power = 10 # power constraint in the weighted sum rate maximization problem 
    noise_power = 1
    path_loss_option = False # used to add a random path loss (drawn from a uniform distribution) to the channel of each user
    path_loss_range = [-5,5] # interval of the uniform distribution from which the path loss if drawn (in dB)
    nr_of_batches_training = 10000 # used for training
    nr_of_batches_test = 10 # used for testing
    nr_of_samples_per_batch = 1
    nr_of_iterations = 1000 # for WMMSE algorithm in Shi et al. 
    nr_of_iterations_nn = 1 # for the deep unfolded WMMSE in our paper

    # User weights in the weighted sum rate (denoted by alpha in our paper)
    user_weights = np.reshape(np.ones(nr_of_users*nr_of_samples_per_batch),(nr_of_samples_per_batch,nr_of_users,1))
    user_weights_for_regular_WMMSE = np.ones(nr_of_users)



    # For repeatability
    # np.random.seed(1234)

    # load samples
    Hall = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(nr_of_batches_test, nr_of_users, nr_of_BS_antennas, 2)).view(np.complex128).squeeze()

    WSR_WMMSE_all =[] # to store the WSR attained by the WMMSE
    MR_WMMSE_all =[] # to store the WSR attained by the WMMSE
    WSR_ZF_all = [] # to store the WSR attained by the zero-forcing 
    WSR_RZF_all = [] # to store the WSR attained by the regularized zero-forcing
    W_WMMSE_all = []

    # power_all = 10**(np.array([0,5,10,15,20])/10)
    power_all = 10**(np.array([15])/10)
    for total_power in power_all:
        print('power =', total_power)
        WSR_WMMSE =[] # to store the WSR attained by the WMMSE
        MR_WMMSE =[] # to store the WSR attained by the WMMSE
        WSR_ZF = [] # to store the WSR attained by the zero-forcing 
        WSR_RZF = [] # to store the WSR attained by the regularized zero-forcing
        WSR_nn = [] # to store the WSR attained by the deep unfolded WMMSE
        training_loss = []
        for i in tqdm(range(nr_of_batches_test)):    
            # print(i)
            batch_for_testing = []
            initial_transmitter_precoder_batch = []
            WSR_WMMSE_batch = 0.0
            MR_WMMSE_batch = 0.0
            WSR_ZF_batch = 0.0
            WSR_RZF_batch = 0.0

            # Building a batch for testing
            channel_realization_regular = Hall[i]

            # Compute the regilarized zero-forcing solution
            RZF_solution = regularized_zero_forcing(channel_realization_regular, total_power)
            WSR_RZF_one_sample = compute_weighted_sum_rate(user_weights_for_regular_WMMSE , channel_realization_regular, RZF_solution, noise_power, scheduled_users)
            WSR_RZF_batch =  WSR_RZF_batch + WSR_RZF_one_sample
            
            # Compute the WMMSE solution
            from time import time
            tstart = time()
            W_WMMSE,_,_,WSR_WMMSE_one_sample, MR_WMMSE_one_sample = run_WMMSE(epsilon, channel_realization_regular, scheduled_users, total_power, noise_power, user_weights_for_regular_WMMSE, nr_of_iterations-1, log=False, init=RZF_solution) 
            WSR_WMMSE_batch =  WSR_WMMSE_batch + WSR_WMMSE_one_sample
            MR_WMMSE_batch =  MR_WMMSE_batch + MR_WMMSE_one_sample
            W_WMMSE_all.append(np.sqrt(total_power)*W_WMMSE/np.linalg.norm(W_WMMSE))
            
            #Testing
            WSR_WMMSE.append(WSR_WMMSE_batch/nr_of_samples_per_batch)
            MR_WMMSE.append(MR_WMMSE_batch/nr_of_samples_per_batch)
            WSR_ZF.append(WSR_ZF_batch/nr_of_samples_per_batch)
            WSR_RZF.append(WSR_RZF_batch/nr_of_samples_per_batch)   


        # print("Training and testing took:", time.time()-start_of_time)
        WSR_WMMSE_all.append(np.mean(WSR_WMMSE))
        MR_WMMSE_all.append(np.mean(MR_WMMSE))
        WSR_ZF_all.append(np.mean(WSR_ZF))
        WSR_RZF_all.append(np.mean(WSR_RZF))
    print("The WSR acheived with the WMMSE algorithm is: ",WSR_WMMSE_all)
    print("The MR acheived with the WMMSE algorithm is: ",MR_WMMSE_all)
    print("The WSR acheived with the regularized zero forcing is: ",WSR_RZF_all)
    # np.save('./W_WMMSE_N=K='+str(nr_of_users)+'.npy', np.array(W_WMMSE_all).reshape((len(power_all), nr_of_batches_test,nr_of_users,nr_of_BS_antennas)))
