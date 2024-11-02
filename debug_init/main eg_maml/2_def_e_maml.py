def experiment(run_params, last_generation=False):
    # Mengambil parameter dari run_params
    seed = int(run_params[0])  # Asumsikan run_params[0] adalah seed
    inner_step_size = float(run_params[1])  # inner step size
    inner_epochs = int(run_params[2])  # inner epochs
    outer_stepsize_reptile = float(run_params[3])  # outer step size for reptile
    outer_stepsize_maml = float(run_params[4])  # outer step size for MAML
    n_iterations = int(run_params[5])  # number of outer updates
    
    run = "E-MAML" if float(run_params[-1]) <= 0.5 else "E-MAML_Synthetic_E-Reptile"  # kondisi if-else
    
    # print('Type of Meta-Learning:', run)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Define task distribution
    n_data_all = 5
    n_sample = n_data_all # minimum 1, maks = n_data_all
    idx_x_all = np.arange(0,n_data_all)[:,None]
    
    # All of the x points data, dengan fitur input pepanjang n_input = 14
    x_all = get_data_test(np.linspace(0, n_data_all-1, np.amin((n_sample, n_data_all), axis=0))[:, None])
    # n_train = 10  # Size of training minibatches
    n_train = 3  # Size of training minibatches, harus < n_data_all
    
    # info_params = f"imax-{n_iterations}-ndata-{n_data_all}-nspl-{n_sample}-ntrain-{n_train}-s-{seed}-iss-{inner_step_size}-ie-{inner_epochs}-osr-{outer_stepsize_reptile}-osm-{outer_stepsize_maml}"
       
    def get_mse_or_loss_val(get_idx_x_all_in):
        x = to_torch(np.array([x_all[i] for i in get_idx_x_all_in]))
        y = to_torch(np.array([y_all[i] for i in get_idx_x_all_in]))

        # cara 1
        model.zero_grad()
        y_pred = model(x)
        individual_losses = (y_pred - y).pow(2).mean(dim=1)  # Loss for each sample along feature dimension
        # print("Individual losses:", individual_losses)

        return individual_losses.data.numpy()
    
    def get_idx_x_all(x_all_in, x_all_to_search):
        idx_result = []

        # Iterate through each element in x_all_in
        for x_in in x_all_in:
            # Iterate through x_all_to_search to find a matching element
            for i, x in enumerate(x_all_to_search):
                # Use np.array_equal to compare arrays element-wise
                if np.array_equal(x_in, x):
                    idx_result.append(i)
                    break  # Stop after the first match is found

        return idx_result

    def gen_task_eg_maml_base_idx_data():
        # elm_model = ELMRegression(n_input, n_hidden1, n_hidden2, n_output)
        # load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg.json', elm_model)  # Ganti dengan nama sheet yang sesuai
        
        elm_model_n_hidden_layers = ELMRegression(n_input, hidden_layers, n_output)
        load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg_6_hidden_layers_100_50_25_12_6_3_31-10-2024-10-09-13.json', elm_model_n_hidden_layers)  # Ganti dengan nama sheet yang sesuai

        # f_randoms_eg_maml akan memproses semua data dalam idx_x_all
        f_randoms_eg_maml = lambda idx_x: np.array(
            [test_single_data_return_pred(elm_model_n_hidden_layers, get_data_test(idx_x_single[0]))
             for idx_x_single in idx_x]
        )

        return f_randoms_eg_maml
    
    def gen_task_eg_maml_base_val_data():
        # elm_model = ELMRegression(n_input, n_hidden1, n_hidden2, n_output)
        # load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg.json', elm_model)  # Ganti dengan nama sheet yang sesuai
        
        elm_model_n_hidden_layers = ELMRegression(n_input, hidden_layers, n_output)
        load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg_6_hidden_layers_100_50_25_12_6_3_31-10-2024-10-09-13.json', elm_model_n_hidden_layers)  # Ganti dengan nama sheet yang sesuai

        # f_randoms_eg_maml akan memproses semua data dalam idx_x_all
        f_randoms_eg_maml = lambda x: np.array(
            [test_single_data_return_pred(elm_model, x_single) for x_single in x]
        )

        return f_randoms_eg_maml
    
    def gen_task(): # sama dengan gen_task_eg_maml_base_val_data()
        
        elm_model_n_hidden_layers = ELMRegression(n_input, hidden_layers, n_output)
        load_model_json(f'model_reg/model_Comb-KMT-Tiny-Reg_6_hidden_layers_100_50_25_12_6_3_31-10-2024-10-09-13.json', elm_model_n_hidden_layers)  # Ganti dengan nama sheet yang sesuai

        # f_randoms_eg_maml akan memproses semua data dalam idx_x_all
        f_randoms_eg_maml = lambda x: np.array(
            [test_single_data_return_pred(elm_model, x_single) for x_single in x]
        )

        return f_randoms_eg_maml

    # Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
    # ==========
    # nn.Sequential: Mudah dan cocok untuk deep learning sederhana yang berurutan, 
    # tidak cocok jika arsitektur membutuhkan koneksi yang kompleks.
    ## -------
    # nn.Linear dalam Kelas nn.Module: Lebih fleksibel dan dapat digunakan untuk arsitektur kompleks, 
    # yang melibatkan banyak hidden layer, skip connections, atau jalur paralel.
    
    # Custom activation function NRReLU
    def NRReLU(x):
        return 1 / (torch.exp(-x) - torch.exp(x))

    activations = [NRReLU, nn.Sigmoid(), nn.Tanh(), nn.ReLU()]  # Custom activations, including NRReLU
    
    # Define a function to create the model with configurable layers and activation functions
    def define_model_type1(n_input, n_hidden_layers, n_output, activations=None):
        layers = []
        input_dim = n_input

        # Ensure activations list matches the number of hidden layers, or use ReLU as default
        if activations is None:
            activations = [F.relu] * len(n_hidden_layers)  # Default to ReLU for all layers
        elif len(activations) != len(n_hidden_layers):
            raise ValueError("Length of activations must match number of hidden layers")

        # Add each hidden layer with the specified number of neurons and activation
        for hidden_units, activation in zip(n_hidden_layers, activations):
            layers.append(nn.Linear(input_dim, hidden_units))
            input_dim = hidden_units  # Update input_dim for the next layer

            # Add the activation layer as a callable function
            layers.append(activation)  # Add the activation function directly

        # Add the final output layer without activation
        layers.append(nn.Linear(input_dim, n_output))

        # Create the model with nn.Sequential
        model = nn.Sequential(*layers)
        return model
    
    def define_model_type2(n_input, n_hidden_layers, n_output):
        #     model = nn.Sequential(
        #         nn.Linear(1, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 1),
        #     )

        layers = []
        input_dim = n_input

        # Add each hidden layer with alternating activation functions
        for i, hidden_units in enumerate(n_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_units))

            # Use ReLU for the first layer, Tanh for others
            if i % 2 == 0:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

            input_dim = hidden_units  # Update input_dim for the next layer

        # Add the final output layer
        layers.append(nn.Linear(input_dim, n_output))

        # Create the model with nn.Sequential
        model = nn.Sequential(*layers)
        return model
    
    # Define sintesis model. Reptile dengan ELMRegressionForReptile - nn.Linear
    model = ModelForSyntheticReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
        
    def save_model_reptile_checkpoint(model, filename):
        # Dapatkan state_dict dari model
        model_state = model.state_dict()

        # Konversi tensor menjadi list untuk serialisasi JSON
        model_state_serializable = {k: v.numpy().tolist() for k, v in model_state.items()}

        # Simpan model ke file JSON
        with open(filename, 'w') as f:
            json.dump(model_state_serializable, f)

    def load_model_reptile_checkpoint(model, filename):
        # Muat model dari file JSON
        with open(filename, 'r') as f:
            model_state_serializable = json.load(f)

        # Konversi kembali dari list ke tensor
        model_state = {k: torch.tensor(np.array(v)) for k, v in model_state_serializable.items()}

        # Memuat state_dict ke model
        model.load_state_dict(model_state)
        model.eval()  # Set model ke mode evaluasi

    def to_torch(x):
        return ag.Variable(torch.Tensor(x))

    # def train_on_batch(x, y):
    #     x = to_torch(x)
    #     y = to_torch(y)
    #     model.zero_grad()
    #     ypred = model(x)
    #     loss = (ypred - y).pow(2).mean()
    #     loss.backward()
    #     for param in model.parameters():
    #         param.data -= inner_step_size * param.grad.data
    
               
    # using ELMRegressionForReptile support param.data dan param.grad.data
    def train_on_batch_eg_maml(x, y):
        x = to_torch(x)
        y = to_torch(y)

        model = ELMRegressionForReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # elm_model_reptile.zero_grad()
        # ypred = test_single_data_return_pred(elm_model_reptile,x)
        # loss = (ypred - y).pow(2).mean()
        # loss.backward()

        # Forward and backward pass
        model.train()
        optimizer.zero_grad()
        ypred = model(x)
        loss = criterion(ypred, y)
        loss.backward()
        optimizer.step()

        for param in model.parameters():
            param.data -= inner_step_size * param.grad.data
            
        # Save model checkpoint
        # filename_ckpt = f'model_reg_ckpt/model_reptile_checkpoint_{datetime.today().astimezone(pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y-%H-%M-%S")}.json'
        # save_model_reptile_checkpoint(model, filename_ckpt)
        
        # return loss.item()  # Optionally return the loss for monitoring
            
    def train_on_batch(x, y):
        x = to_torch(x)
        y = to_torch(y)

        model = ELMRegressionForReptile(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # elm_model_reptile.zero_grad()
        # ypred = test_single_data_return_pred(elm_model_reptile,x)
        # loss = (ypred - y).pow(2).mean()
        # loss.backward()

        # Forward and backward pass
        model.train()
        optimizer.zero_grad()
        ypred = model(x)
        loss = criterion(ypred, y)
        loss.backward()
        optimizer.step()

        for param in model.parameters():
            param.data -= inner_step_size * param.grad.data
            
        # Save model checkpoint
        # filename_ckpt = f'model_reg_ckpt/model_reptile_checkpoint_{datetime.today().astimezone(pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y-%H-%M-%S")}.json'
        # save_model_reptile_checkpoint(model, filename_ckpt)
        
        # return loss.item()  # Optionally return the loss for monitoring
        
    # Cara Memuat model dari checkpoint
    #     try:
    #         load_model_reptile_checkpoint(model, filename_ckpt)
    #         print("Model berhasil dimuat dari:", filename_ckpt)
    #     except Exception as e:
    #         print("Terjadi kesalahan saat memuat model:", e)

    #     # Sekarang Anda bisa menggunakan model untuk melakukan prediksi atau melanjutkan pelatihan
    #     # Contoh prediksi
    #     x_test = to_torch([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])  # Ganti dengan data yang sesuai
    #     model.eval()  # Set model ke mode evaluasi
    #     with torch.no_grad():
    #         prediction = model(x_test)
    #         print("Hasil prediksi:", prediction.numpy())

    def predict(x):
        x = to_torch(x)
        return model(x).data.numpy()
    
    #     def save_info_params(
    #         n_iterations, n_data_all, n_sample, n_train, seed, inner_step_size,
    #         inner_epochs, outer_stepsize_reptile, outer_stepsize_maml,
    #         run, final_lossval, filename_last_Model
    #     ):
    #         # Construct the info_params dictionary
    #         info_params = {
    #             "imax": n_iterations,
    #             "ndata": n_data_all,
    #             "nspl": n_sample,
    #             "ntrain": n_train,
    #             "s": seed,
    #             "iss": inner_step_size,
    #             "ie": inner_epochs,
    #             "osr": outer_stepsize_reptile,
    #             "run": run,
    #             "osm": outer_stepsize_maml,
    #             "final_lossval": float(final_lossval)
    #         }

    #         # Construct the filename with all the specified information
    #         filename = (
    #             f"model_reg_last/model_params_last_{run}_{final_lossval:.3f}_"
    #             f"{filename_last_Model}.json"
    #         )

    #         # Ensure the directory exists
    #         os.makedirs(os.path.dirname(filename), exist_ok=True)

    #         # Save the info_params dictionary to the file as JSON
    #         with open(filename, 'w') as json_file:
    #             json.dump(info_params, json_file, indent=4)

        # print(f"Parameters saved to {filename}")
        # return filename

    # Choose a fixed task and minibatch for visualization
    f_plot = gen_task()
    # xtrain_plot = x_all[rng.choice(len(x_all), size=n_train)]
    xtrain_plot = np.array([x_all[i] for i in rng.choice(len(x_all), size=n_train)])

    # plt.cla()
    # Set figure and axis properties
    # plt.figure()
    
    # Set gray background color
    # plt.gca().set_facecolor('#f0f0f0')  # Light gray color for the background
    # plt.gca().set_facecolor('lightgray')
    
    # Add grid lines
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Training loop
    filename_first_n_last_Loss = datetime.today().astimezone(pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y-%H-%M-%S")
    for iteration in range(n_iterations):
        weights_before = deepcopy(model.state_dict())

        # Generate task
        f = gen_task()
        y_all = f(x_all)

        # Do SGD on this task
        inds = rng.permutation(len(x_all))
        train_ind = inds[:-1 * n_train]
        val_ind = inds[-1 * n_train:]       # Val contains 1/5th of the gt model (com. model)

        for _ in range(inner_epochs):
            for start in range(0, len(train_ind), n_train):
                mbinds = train_ind[start:start + n_train]
                # print('mbinds =', mbinds)
                # print()
                # print('x_all[mbinds] =', x_all[mbinds])
                # print()
                # print('y_all[mbinds] =', y_all[mbinds])
                x_all_mbinds = np.array([x_all[i] for i in mbinds])
                y_all_mbinds = np.array([y_all[i] for i in mbinds])
                # train_on_batch(x_all[mbinds], y_all[mbinds])
                train_on_batch(x_all_mbinds, y_all_mbinds)
                
                # print('=======================')

        if run == 'E-MAML':
            outer_step_size = outer_stepsize_maml * (1 - iteration / n_iterations)  # linear schedule
            for start in range(0, len(val_ind), n_train):
                dpinds = val_ind[start:start + n_train]
                # print('dpinds =', dpinds)
                # print()
                
                # x = to_torch(x_all[dpinds])
                x = to_torch(np.array([x_all[i] for i in dpinds]))
                
                # y = to_torch(y_all[dpinds])
                y = to_torch(np.array([y_all[i] for i in dpinds]))

                # Compute the grads
                model.zero_grad()
                y_pred = model(x)
                loss = (y_pred - y).pow(2).mean()
                loss.backward()
                

                # Reload the model
                model.load_state_dict(weights_before)

                # SGD on the params
                for param in model.parameters():
                    param.data -= outer_step_size * param.grad.data
            # print(weights_before)
        else:
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient
            weights_after = model.state_dict()
            outerstepsize = outer_stepsize_reptile * (1 - iteration / n_iterations)  # linear schedule
            model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                                   for name in weights_before})

        # Periodically plot the results on a particular task and minibatch
        # if (plot and ((iteration == 0) or ((iteration + 1) % 1000 == 0))):
#         if (plot and ((iteration == 0) or ((iteration + 1) % n_iterations == 0))):
#             plt.cla()
#             # plt.cla()
            
#             # Set gray background color
#             # plt.gca().set_facecolor('#f0f0f0')  # Light gray color for the background
            
#             # Set gray background color
#             # plt.gca().set_facecolor('#f0f0f0')  # Light gray color for the background
#             plt.gca().set_facecolor('lightgray')

#             # Add grid lines
#             plt.grid(color='gray', linestyle='--', linewidth=0.5)
            
#             f = f_plot
#             weights_before = deepcopy(model.state_dict())  # save snapshot before evaluation
            
#             # plt.plot(x_all, predict(x_all), label="pred after 0", color=(0, 0, 1))
#             get_idx_x_all_to_2d_plot = get_idx_x_all(x_all, x_all) # agar dapat diplot pd 2D
#             get_mse_or_loss_val_to_2d_plot = get_mse_or_loss_val(get_idx_x_all(x_all, x_all))
#             plt.plot(get_idx_x_all_to_2d_plot, get_mse_or_loss_val_to_2d_plot, label="pred after 0", color=(0, 0, 1))
            
#             for inneriter in range(32):
#                 train_on_batch(xtrain_plot, f(xtrain_plot))
#                 if (inneriter + 1) % 8 == 0:
#                     frac = (inneriter + 1) / 32
#                     # plt.plot(x_all, predict(x_all), label="pred after %i" % (inneriter + 1), color=(frac, 0, 1 - frac))
                    
#                     get_idx_x_all_to_2d_plot = get_idx_x_all(x_all, x_all) # agar dapat diplot pd 2D
#                     get_mse_or_loss_val_to_2d_plot = get_mse_or_loss_val(get_idx_x_all(x_all, x_all))
#                     plt.plot(get_idx_x_all_to_2d_plot, get_mse_or_loss_val_to_2d_plot, label="pred after %i" % (inneriter + 1), color=(frac, 0, 1 - frac))
            
#             # plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
#             # plt.plot(x_all, f(x_all), label="ground truth from sin(x)", color=(0, 1, 0))
            
#             get_idx_x_all_to_2d_plot = get_idx_x_all(x_all, x_all) # agar dapat diplot pd 2D
#             get_mse_or_loss_val_to_2d_plot = get_mse_or_loss_val(get_idx_x_all(x_all, x_all))
#             plt.plot(get_idx_x_all_to_2d_plot, get_mse_or_loss_val_to_2d_plot, label="ground truth from comb. model", color=(0, 1, 0))
            
            
#             lossval = np.square(predict(x_all) - f(x_all)).mean()
#             # plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
            
#             # print("xtrain_plot: ",xtrain_plot)
            
#             get_idx_x_all_to_2d_plot = get_idx_x_all(xtrain_plot, x_all) # agar dapat diplot pd 2D
#             get_mse_or_loss_val_to_2d_plot = get_mse_or_loss_val(get_idx_x_all_to_2d_plot)
            
#             # print("idx xtrain_plot: ",get_idx_x_all_to_2d_plot)
        
            
#             plt.plot(get_idx_x_all_to_2d_plot, get_mse_or_loss_val_to_2d_plot, "x", label="train", color="k")
            
#             plt.ylim(-4, 4)
#             plt.xlim(0, 4)  # Set x-axis limits
#             plt.xticks(range(5))  # Set x-ticks to show 0, 1, 2, 3, 4
#             plt.xlabel("index of data")  # Label for x-axis
#             plt.ylabel("loss value")     # Label for y-axis
#             plt.legend(loc="lower right")
            
#             plt.savefig(f"loss_e_maml/plot_{run}_{lossval:.3f}_iter_{iteration}_{filename_first_n_last_Loss}.png")
#             plt.savefig(f"loss_e_maml/plot_{run}_{lossval:.3f}_iter_{iteration}_{filename_first_n_last_Loss}.pdf")

            
#             plt.pause(0.01)
#             model.load_state_dict(weights_before)  # restore from snapshot
#             print(f"-----------------------------")
#             print(f"iteration               {iteration + 1}")
#             print(f"loss on plotted curve   {lossval:.3f}")  # would be better to average loss over a set of examples, but this is optimized for brevity

    
    # print()
    final_lossval = np.square(predict(x_all) - f(x_all)).mean()
    # print(f"final loss on last model = {final_lossval:.3f}") 
    filename_last_Model = datetime.today().astimezone(pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y-%H-%M-%S")
    
    # Save last loss
    # plt.savefig(f"loss_e_maml/plot_e_maml_{filename_last_Model_n_Loss}.png")
    # plt.savefig(f"loss_e_maml/plot_e_maml{filename_last_Model_n_Loss}.pdf")
    
    # Save the plot as PNG and PDF only after plotting
    # plt.savefig(f"loss_e_maml/plot_e_maml_{filename_last_Model_n_Loss}.png")
    # plt.savefig(f"loss_e_maml/plot_e_maml_{filename_last_Model_n_Loss}.pdf")

    # Optionally, show the plot if you want to display it interactively
    # plt.show()  # Use this only if you want to display the plot interactively
    
    
    # Save last model checkpoint
    # path_last_Model = f'model_reg_last/model_last_{run}_{final_lossval:.3f}_{filename_last_Model}.json'
    path_last_Model = f'model_reg_last/model_last_{run}_{final_lossval:.3f}_{filename_last_Model}'
    # save_model_reptile_checkpoint(model, path_last_Model)
    # save_info_params(
    #     n_iterations, n_data_all, n_sample, n_train, seed, inner_step_size,
    #     inner_epochs, outer_stepsize_reptile, outer_stepsize_maml,
    #     run, final_lossval, filename_last_Model)
    
    config = {
        "n_iterations": n_iterations,
        "n_data_all": n_data_all,
        "n_sample": n_sample,
        "n_train": n_train,
        "seed": seed,
        "inner_step_size": inner_step_size,
        "inner_epochs": inner_epochs,
        "outer_stepsize_reptile": outer_stepsize_reptile,
        "outer_stepsize_maml": outer_stepsize_maml,
        "run": run,
        "final_lossval": final_lossval,
        "filename_last_Model": filename_last_Model,
        "path_filename_last_Model": path_last_Model
        
    }
    if(last_generation==True):
        return final_lossval, path_last_Model, model, config
    else:
        return final_lossval