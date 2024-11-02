def gtvga(objective_func, bounds, num_generations=num_iterations_all, population_size=pop_size_all, crossover_prob=0.7, mutation_prob=0.1, elite_percentage=1., full_logger=1):
    # Inisialisasi populasi awal dengan torch
    bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
    # population = torch.rand(population_size, len(bounds)).to(bounds_tensor.device) * (bounds_tensor[:, 1] - bounds_tensor[:, 0]) + bounds_tensor[:, 0]
    
    # Constants
    c1i, c1f = 2.5, 0.5
    c2i, c2f = 0.5, 2.5
    cr, mr = crossover_prob, mutation_prob
    
    dim = len(bounds)
    
    # Initialize population within the specified bounds
    population = torch.rand((population_size, dim), device=my_device) * \
    (torch.tensor(bounds, device=my_device)[:, 1] - \
     torch.tensor(bounds, device=my_device)[:, 0]) + \
    torch.tensor(bounds, device=my_device)[:, 0]
    
    # print(population)
    
    # Inisialisasi variabel sementara
    temp_path_last_Model_at_gtvga = None
    temp_model_at_gtvga = None
    temp_config_at_gtvga = None
    temp_i_ind_at_gtvga = None
    temp_best_fitness_in_gtvga = None
    last_generation = False
    
    # Initialize untuk grafik konvergensi
    max_gbest_each_iter = torch.empty(num_generations, device=my_device) # makna max disini tdk selalu nilai Max, tetapi krn loss, maka nilai yg paling min
    mean_gbest_each_iter = torch.empty(num_generations, device=my_device)
    
    # Fungsi untuk mengevaluasi fitness
    # def evaluate_fitness(pop):
    #     return torch.tensor([objective_func(ind) for ind in pop], dtype=torch.float32)

    # Melakukan proses GA
    for generation in range(num_generations):
        
        print('generation -> ',generation)
        
        if(generation==(num_generations-1)):
            last_generation = True
        
        # Adaptive crossover and mutation rates utilize geometric time variant (GTV) 
        cr_ = ((c2f - c2i) * (generation / num_generations)) + c2i
        mr_ = ((c1f - c1i) * (generation / num_generations)) + c1i
        
        crossover_prob = cr_
        mutation_prob = mr_
        
        # Langkah 1: Crossover untuk menghasilkan offspring
        offspring_crossover = []
        for _ in range(population_size):
            parent1, parent2 = population[torch.randint(0, population_size, (1,)).item()], population[torch.randint(0, population_size, (1,)).item()]
            if torch.rand(1).item() < crossover_prob:
                crossover_point = torch.randint(1, len(bounds) - 1, (1,)).item()
                child = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
            else:
                child = parent1  # Tidak crossover
            offspring_crossover.append(child)
        offspring_crossover = torch.stack(offspring_crossover)

        # Langkah 2: Mutasi untuk menghasilkan offspring
        offspring_mutation = []
        for ind in offspring_crossover:
            if torch.rand(1).item() < mutation_prob:
                mutation_point = torch.randint(0, len(bounds), (1,)).item()
                mutated_ind = ind.clone()
                mutated_ind[mutation_point] = torch.rand(1).item() * (bounds_tensor[mutation_point, 1] - bounds_tensor[mutation_point, 0]) + bounds_tensor[mutation_point, 0]
                offspring_mutation.append(mutated_ind)
            else:
                offspring_mutation.append(ind)  # Tidak mutasi
        offspring_mutation = torch.stack(offspring_mutation)

        # Langkah 3: Gabungkan populasi awal dan offspring
        combined_population = torch.cat((population, offspring_crossover, offspring_mutation), dim=0)

        # Langkah 4: Seleksi elitisme
        # fitness_values = evaluate_fitness(combined_population)
        fitness_values = objective_func(combined_population)
        # fitness_values, _, _, _, _ = objective_func(combined_population)
        
        elite_count = int(population_size * elite_percentage)
        elite_indices = torch.argsort(fitness_values)[:elite_count]
        population = combined_population[elite_indices]

        # Optional: Logging
        # if full_logger and (generation % 10 == 0 or generation == num_generations - 1):
        #     best_fitness = torch.min(fitness_values[elite_indices])
        #     print(f"Generation {generation}: Best Fitness = {best_fitness.item()}")
            
        # Get the best solution found
        # all_fitness_new_population = objective_func(population)
        # all_fitness_new_population, path_last_Model_in_gtvga, model_in_gtvga, config_in_gtvga, i_ind_in_gtvga = objective_func(population)
        
        if(last_generation):
            all_fitness_new_population, path_last_Model_in_gtvga, model_in_gtvga, config_in_gtvga, i_ind_in_gtvga = objective_func(population, True)
        else:
            all_fitness_new_population = objective_func(population)
        
        # best_index = torch.argmin(fitness_values)
        # best_index = torch.argmin(fitness_values[elite_indices])
        best_index = torch.argmin(all_fitness_new_population)
        # best_solution = population[best_index]
        best_solution = population[best_index]
        # best_fitness = fitness_values[best_index]
        best_fitness = all_fitness_new_population[best_index]
        
        # untuk membuat grafik konvergensi
        max_gbest_each_iter[generation] = best_fitness
        mean_gbest_each_iter[generation] = fitness_values.mean()
        
        # if generation == 0:
        #     # Simpan path_last_Model, model, dan config pada variabel sementara untuk pertama kali
        #     temp_path_last_Model_in_gtvga = path_last_Model_in_gtvga
        #     temp_model_in_gtvga = model_in_gtvga
        #     temp_config_in_gtvga = config_in_gtvga
        #     temp_i_ind_in_gtvga = i_ind_in_gtvga
        #     temp_best_fitness_in_gtvga = best_fitness
        # else:
        #     # Jika fitness dari individu ke-i lebih kecil, replace variabel sementara
        #     if best_fitness < temp_best_fitness_in_gtvga:
        #         temp_path_last_Model_in_gtvga = path_last_Model_in_gtvga
        #         temp_model_in_gtvga = model_in_gtvga
        #         temp_config_in_gtvga = config_in_gtvga
        #         temp_i_ind_in_gtvga = i_ind_in_gtvga
        #         temp_best_fitness_in_gtvga = best_fitness

    # Mengembalikan populasi terbaik dan nilai fitness terbaik
    # best_fitness = objective_func(population)
    # best_solution_index = torch.argmin(best_fitness)
    
    # return population[best_solution_index], best_fitness[best_solution_index]
    
    temp_path_last_Model_in_gtvga = path_last_Model_in_gtvga
    temp_model_in_gtvga = model_in_gtvga
    temp_config_in_gtvga = config_in_gtvga
    temp_i_ind_in_gtvga = i_ind_in_gtvga
    temp_best_fitness_in_gtvga = best_fitness

    if(full_logger == None):
        return best_solution, best_fitness, population
    else:
        # return torch.tensor(xp_by_linspace, device=my_device), \
        #        torch.tensor(xp_by_linspace_to_p, device=my_device), log, f_new
        # return X_ori, Y_ori, xp_by_non_or_with_linspace, \
        #        xp_by_non_or_with_linspace_to_p, log, log_TARty, f_new
        # return best_solution, best_fitness[best_solution_index], population[best_solution_index], \
        #            max_gbest_each_iter, mean_gbest_each_iter
        return best_solution, best_fitness, population, \
                   max_gbest_each_iter, mean_gbest_each_iter, \
                    temp_path_last_Model_in_gtvga, temp_model_in_gtvga, temp_config_in_gtvga, temp_i_ind_in_gtvga, temp_best_fitness_in_gtvga