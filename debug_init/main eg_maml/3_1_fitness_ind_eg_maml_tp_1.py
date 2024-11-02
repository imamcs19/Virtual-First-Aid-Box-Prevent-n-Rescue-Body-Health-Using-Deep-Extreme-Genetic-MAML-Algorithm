def fitness_ind_eg_maml_tp_1(x=None, last_generation=False):
    # Tentukan batas untuk variabel
    bounds_tp_1 = [
        (0, 4),  # Bound untuk seed (misalnya dari 0 sampai 100)
        (0.01, 0.1),  # Bound untuk inner_step_size (misalnya dari 0.01 sampai 0.1)
        (1, 7),  # Bound untuk inner_epochs (misalnya dari 1 sampai 10)
        (0.01, 0.2),  # Bound untuk outer_stepsize_reptile (misalnya dari 0.01 sampai 0.2)
        (0.001, 0.05),  # Bound untuk outer_stepsize_maml (misalnya dari 0.001 sampai 0.05)
        (1, 10),  # Bound untuk n_iterations (misalnya dari 1 sampai 50)
        (0, 1)  # Bound untuk run type (misalnya dari 0 sampai 1) "E-MAML" if <= 0.5 else "E-MAML_Synthetic_E-Reptile"
    ]
    
    def calculate_fitness(run_params_in, last_generation_in=False):
        # Asumsikan 'run' adalah tensor yang berisi parameter individu
        # Konversi 'run' ke format yang sesuai untuk experiment
        # Misalnya, kita dapat mengonversi tensor menjadi list atau numpy array, tergantung implementasi experiment
        run_params = run_params_in.detach().cpu().numpy().tolist()  # Jika menggunakan PyTorch, ambil data ke CPU dan konversi ke list

        # Memanggil fungsi experiment dengan parameter yang sesuai
        # fitness_value = experiment(run_params)  # plot=False untuk tidak menampilkan plot
        # fitness_value, path_last_Model, model, config = experiment(run_params)  # plot=False untuk tidak menampilkan plot
                
        if(last_generation_in == True):
            fitness_value, path_last_Model, model, config = experiment(run_params, True)  # plot=False untuk tidak menampilkan plot
            return fitness_value, path_last_Model, model, config
        else:
            # print()
            fitness_value = experiment(run_params)
            return fitness_value

        # Mengembalikan nilai fitness
        # return fitness_value
        # return fitness_value, path_last_Model, model, config

    # Jika x tidak diberikan, inisialisasi dengan nilai acak
    if x is None:
        pop_size_all = 5
        
        dim = len(bounds_tp_1)
        num_ind = pop_size_all
        
        x = torch.rand((num_ind, dim), device=my_device) * \
            (torch.tensor(bounds_tp_1, device=my_device)[:, 1] - \
             torch.tensor(bounds_tp_1, device=my_device)[:, 0]) + \
            torch.tensor(bounds_tp_1, device=my_device)[:, 0]
        
        # print(x)
    else:
        # Menyesuaikan tipe data tensor x
        if check_dtype_support(torch.float64):
            x = x.to(torch.float64)
            # print('utilize torch.float64')
        else:
            x = x.to(torch.float32)
            # print('utilize torch.float32')
            
        
    
    # Mendapatkan ukuran populasi
    get_pop_size = x.shape[0]
    
    fitness_all = []  # tampung semua nilai fitness dari individu
    
    # Inisialisasi variabel sementara
    temp_path_last_Model = None
    temp_model = None
    temp_config = None
    temp_i_ind = None
    
    for i_ind in range(get_pop_size):
        # print(x[i_ind])
        # Hitung fitness untuk individu ke-i
        # fitness = calculate_fitness(x[i_ind])  # Misalkan ada fungsi calculate_fitness
        # fitness, path_last_Model, model, config = calculate_fitness(x[i_ind])  # Misalkan ada fungsi calculate_fitness
        
        if(last_generation==True):
            fitness, path_last_Model, model, config = calculate_fitness(x[i_ind], True)  # Misalkan ada fungsi calculate_fitness
            
            if i_ind == 0:
                # Simpan path_last_Model, model, dan config pada variabel sementara untuk pertama kali
                temp_path_last_Model = path_last_Model
                temp_model = model
                temp_config = config
                temp_i_ind = i_ind
            else:
                # Jika fitness dari individu ke-i lebih kecil, replace variabel sementara
                if fitness < fitness_all[i_ind - 1]:
                    temp_path_last_Model = path_last_Model
                    temp_model = model
                    temp_config = config
                    temp_i_ind = i_ind
        else:
            fitness = calculate_fitness(x[i_ind])  # Misalkan ada fungsi calculate_fitness
            
        fitness_all.append(fitness)
        
        # if i_ind == 0:
        #     # Simpan path_last_Model, model, dan config pada variabel sementara untuk pertama kali
        #     temp_path_last_Model = path_last_Model
        #     temp_model = model
        #     temp_config = config
        #     temp_i_ind = i_ind
        # else:
        #     # Jika fitness dari individu ke-i lebih kecil, replace variabel sementara
        #     if fitness < fitness_all[i_ind - 1]:
        #         temp_path_last_Model = path_last_Model
        #         temp_model = model
        #         temp_config = config
        #         temp_i_ind = i_ind

    # return torch.tensor(fitness_all, device=my_device)
    # return torch.tensor(fitness_all, device=my_device), temp_path_last_Model, temp_model, temp_config, temp_i_ind
    if(last_generation==True):
        return torch.tensor(fitness_all, device=my_device), temp_path_last_Model, temp_model, temp_config, temp_i_ind
    else:
        return torch.tensor(fitness_all, device=my_device)