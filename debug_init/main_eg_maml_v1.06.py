def run_experiments(test_functions, optimization_algorithms, bounds_test_functions):
    results = []
    # for func in test_functions:
    for func,bound in zip(test_functions,bounds_test_functions):
        func_results = {'function': func.__name__, 'results': []}
        for optimizer in optimization_algorithms:
            print(func.__name__)
            print(bound)
            print(optimizer.__name__)
            print()
            
            start_time = time.time()
            # best_solution, best_fitness, all_solutions = optimizer(func, bounds)
            
            # return global_best_position, global_best_fitness, particles_position, \
            #        max_gbest_each_iter, mean_gbest_each_iter
            
            # best_solution_gpu, best_fitness_gpu, all_solutions_gpu, \
            # max_gbest_each_iter_gpu, mean_gbest_each_iter_gpu = optimizer(func, bound)
            
            best_solution_gpu, best_fitness_gpu, all_solutions_gpu, \
            max_gbest_each_iter_gpu, mean_gbest_each_iter_gpu, \
            final_path_last_Model, final_model, final_config, final_i_ind, final_best_fitness = optimizer(func, bound)
            
            # final_path_last_Model = f'model_reg_last/model_last_{run}_{final_lossval:.3f}_{filename_last_Model}'
            
            save_last_model_reptile_checkpoint(final_model, final_path_last_Model+'_'+func.__name__+'_'+optimizer.__name__+'.json')
            # save_last_info_params(final_config["n_iterations"], final_config["n_data_all"], final_config["n_sample"], \
            #                       final_config["n_train"], final_config["seed"], final_config["inner_step_size"], \
            #                       final_config["inner_epochs"], final_config["outer_stepsize_reptile"], \
            #                       final_config["outer_stepsize_maml"], final_config["run"], final_config["final_lossval"], \
            #                       final_config["filename_last_Model"])
            save_last_info_params(
                final_config["n_iterations"], final_config["n_data_all"], final_config["n_sample"],
                final_config["n_train"], final_config["seed"], final_config["inner_step_size"],
                final_config["inner_epochs"], final_config["outer_stepsize_reptile"],
                final_config["outer_stepsize_maml"], final_config["run"], final_config["final_lossval"],
                final_config["filename_last_Model"]+'_'+func.__name__+'_'+optimizer.__name__,
                final_config["path_filename_last_Model"]+'_'+func.__name__+'_'+optimizer.__name__+'.json'
            )
            
            print(final_config)
            print()

            
            best_solution, best_fitness, all_solutions, \
            max_gbest_each_iter, mean_gbest_each_iter = \
            best_solution_gpu.cpu().numpy().flatten(),\
            best_fitness_gpu.cpu().numpy().flatten(),\
            all_solutions_gpu.cpu().numpy().flatten(),\
            max_gbest_each_iter_gpu.cpu().numpy().flatten(),\
            mean_gbest_each_iter_gpu.cpu().numpy().flatten()
            
            # my_device = 'cpu'
            running_time = time.time() - start_time
            # fitness_values_gpu, _, _, _, _  = func(all_solutions_gpu)
            fitness_values_gpu = func(all_solutions_gpu)
            # fitness_values = fitness_values_gpu.cpu().numpy().flatten()
            fitness_values = fitness_values_gpu.cpu().numpy().flatten()

            func_results['results'].append({
                'algorithm_name': optimizer.__name__,
                'best_solution': best_solution,
                'best_fitness': best_fitness,
                'median_fitness': np.median(fitness_values),
                'worst_fitness': np.max(fitness_values),
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'running_time': running_time,
                'max_gbest_each_iter': max_gbest_each_iter,
                'mean_gbest_each_iter': mean_gbest_each_iter
            })

        results.append(func_results)

    return results


test_functions = [fitness_ind_eg_maml_tp_1, fitness_ind_eg_maml_tp_2]

alias_func_name = ['eg_maml_tiny_inv', 'eg_maml_middle_inv']

# optimization_algorithms = [ga]
optimization_algorithms = [ga, gtvga]
# optimization_algorithms = [ga, gtvga, ptvpso]

bounds_tp_1 = [
        (0, 4),  # Bound untuk seed (misalnya dari 0 sampai 100)
        (0.01, 0.1),  # Bound untuk inner_step_size (misalnya dari 0.01 sampai 0.1)
        (1, 7),  # Bound untuk inner_epochs (misalnya dari 1 sampai 10)
        (0.01, 0.2),  # Bound untuk outer_stepsize_reptile (misalnya dari 0.01 sampai 0.2)
        (0.001, 0.05),  # Bound untuk outer_stepsize_maml (misalnya dari 0.001 sampai 0.05)
        (1, 10),  # Bound untuk n_iterations (misalnya dari 1 sampai 50)
        (0, 1)  # Bound untuk run type (misalnya dari 0 sampai 1) "E-MAML" if <= 0.5 else "E-MAML_Synthetic_E-Reptile"
    ]

bounds_tp_2 = [
        (0, 8),  # Bound untuk seed (misalnya dari 0 sampai 100)
        (0.001, 0.2),  # Bound untuk inner_step_size (misalnya dari 0.01 sampai 0.1)
        (1, 14),  # Bound untuk inner_epochs (misalnya dari 1 sampai 10)
        (0.001, 0.4),  # Bound untuk outer_stepsize_reptile (misalnya dari 0.01 sampai 0.2)
        (0.0001, 0.1),  # Bound untuk outer_stepsize_maml (misalnya dari 0.001 sampai 0.05)
        (1, 20),  # Bound untuk n_iterations (misalnya dari 1 sampai 50)
        (0, 1)  # Bound untuk run type (misalnya dari 0 sampai 1) "E-MAML" if <= 0.5 else "E-MAML_Synthetic_E-Reptile"
    ]


bounds_test_functions = [bounds_tp_1, bounds_tp_2]

# results = run_experiments(test_functions, optimization_algorithms, bounds)
results = run_experiments(test_functions, optimization_algorithms, bounds_test_functions)

# Menampilkan hasil dalam bentuk tabel
def display_results_table(results):
    print("\nResults:")
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("| Function           | Algorithm      | Best Solution                   | Best Fitness | Median Fitness | Worst Fitness | Mean Fitness | Std Fitness | Running Time (s) |")
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
    for func_result in results:
        for result in func_result['results']:
            print(f"| {func_result['function']:<20} | {result['algorithm_name']:<15} | {result['best_solution']} | "
                  f"{result['best_fitness']} | {result['median_fitness']:<15} | {result['worst_fitness']:<13} | "
                  f"{result['mean_fitness']:<12} | {result['std_fitness']:<11} | {result['running_time']:<17.5f} |")
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------")

def display_results_table_df(results):
    # Membuat list untuk menampung data
    data = []
    
    # Loop melalui setiap fungsi hasil
    for func_result in results:
        for result in func_result['results']:
            # Menambahkan data dalam bentuk dictionary ke dalam list
            data.append({
                'Function': func_result['function'],
                'Algorithm': result['algorithm_name'],
                'Best Solution': result['best_solution'],
                'Best Fitness': result['best_fitness'],
                'Median Fitness': result['median_fitness'],
                'Worst Fitness': result['worst_fitness'],
                'Mean Fitness': result['mean_fitness'],
                'Std Fitness': result['std_fitness'],
                'Running Time (s)': result['running_time']
            })
    
    # Membuat DataFrame dari data
    df = pd.DataFrame(data)
    
    # Menampilkan DataFrame
    print("\nResults:")
    # display(df)
    # display(df.style.hide_index())
    display(df.style.hide(axis='index'))
    
    # return df

# Function to save results to CSV, Excel, and JSON files
def save_results_to_file(results, file_format, path_to_save=None):
    for func_result in results:
        df = pd.DataFrame(func_result['results'])
        
        # Define file path based on function name and file format
        if path_to_save is None:
            filename = f"{func_result['function']}_{file_format}_results"
        else:
            filename = f"{path_to_save}/{func_result['function']}_{file_format}_results"
        
        # Save in CSV, Excel, and JSON formats
        df.to_csv(f"{filename}.csv", index=False)
        df.to_excel(f"{filename}.xlsx", index=False)
        df.to_json(f"{filename}.json", orient="records", indent=4)

# Function to create a chart and save it to a PDF file
def save_chart_to_pdf(results,path_to_save=None):
    if path_to_save == None:
        pdf_pages = PdfPages("chart_fitness.pdf")
    else:
        pdf_pages = PdfPages(path_to_save+"/chart_fitness.pdf")        
        
    # patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    patterns = [ "\\" , "." , "|" , "-" , "+" , "x", "o", "O", "/", "*" ]

    # ax1 = fig.add_subplot(111)
    # for i in range(len(patterns)):
    #     ax1.bar(i, 3, color='green', edgecolor='black', hatch=patterns[i])

    nama_all_alg = [results[0]['results'][i]['algorithm_name'] for i in range(len(results[0]['results']))]
    len_nama_all_alg = len(nama_all_alg)
    # print(nama_all_alg)
    
    # untuk plotting hasil nilai fitness final:
    # ----------------------------------
    for func_result in results:
        plt.figure(figsize=(10, 6))
        
        for j in range(len_nama_all_alg):
            # print(j)
            plt.bar(func_result['results'][j]['algorithm_name'], \
                    func_result['results'][j]['best_fitness'], \
                    label='Best Fitness', edgecolor='white', \
                    hatch=patterns[-j])
        
        
        plt.title(f"Best Fitness for {func_result['function']}")
        plt.xlabel("Algorithm")
        plt.ylabel("Fitness Value")
        plt.legend()
        pdf_pages.savefig()
        # pdf_pages.savefig(bbox_inches='tight', dpi=1000)
        
        plt.close()
        
    # untuk plotting proses pergerakan konvergensi dari hasil nilai fitness final:
    # ----------------------------------
    for func_result in results:
        # fig, ax = plt.figure(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        
        number_color = len_nama_all_alg
        # cmap = plt.get_cmap('gnuplot')
        cmap = plt.get_cmap('jet')
        colors = [cmap(i) for i in np.linspace(0, 1, number_color)]
        
        for j in range(len_nama_all_alg):
            plt.plot(np.arange(1,num_iterations_all+1), func_result['results'][j]['max_gbest_each_iter'], color = colors[j], label=func_result['results'][j]['algorithm_name'].upper())      
        
        # plt.add_subplot(111).set_xticks(arange(1,3,0.5)) # You can actually compute the interval You need - and substitute here
        # ax.set_xticks(arange(1,num_iterations_all+1,1)) # You can actually compute the interval You need - and substitute here
        plt.title(f"Convergence of Max. Fitness Value for {func_result['function']}")
        plt.xlabel("Number of Iteration")
        plt.ylabel("Fitness Value")
        plt.legend()
        pdf_pages.savefig()
        # pdf_pages.savefig(bbox_inches='tight', dpi=1000)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        for j in range(len_nama_all_alg):
            plt.plot(np.arange(1,num_iterations_all+1),func_result['results'][j]['mean_gbest_each_iter'],color = colors[j],label=func_result['results'][j]['algorithm_name'].upper())
        
        
        plt.title(f"Convergence of Mean Fitness Value for {func_result['function']}")
        plt.xlabel("Number of Iteration")
        plt.ylabel("Fitness Value")
        plt.legend()
        pdf_pages.savefig()
        # pdf_pages.savefig(bbox_inches='tight', dpi=1000)
        plt.close()


    pdf_pages.close()
    
# Analisis statistik p-values antar algoritma
# def statistical_analysis_with_visualization(results, suffix, folder_name):
#     for i in range(len(results)):
#         function_name = results[i]['function']
#         algorithm_results = results[i]['results']
        
#         # Create a dictionary to hold fitness values
#         fitness_data = {}
        
#         # Collect max_gbest_each_iter for each algorithm
#         for algorithm in algorithm_results:
#             algorithm_name = algorithm['algorithm_name']
#             fitness_data[algorithm_name] = algorithm['max_gbest_each_iter']  # or use mean_gbest_each_iter

#         # Perform pairwise statistical analysis between algorithms
#         algorithm_names = list(fitness_data.keys())
#         for j in range(len(algorithm_names)):
#             for k in range(j + 1, len(algorithm_names)):
#                 algorithm1 = algorithm_names[j]
#                 algorithm2 = algorithm_names[k]
                
#                 fitness_values1 = fitness_data[algorithm1]
#                 fitness_values2 = fitness_data[algorithm2]

#                 # Check for enough data points and variability
#                 if len(fitness_values1) < 2 or len(fitness_values2) < 2:
#                     print(f"Not enough data to perform t-test for {function_name} using {algorithm1} and {algorithm2}.")
#                     continue
                
#                 std1, std2 = np.std(fitness_values1, ddof=1), np.std(fitness_values2, ddof=1)

#                 if std1 == 0 or std2 == 0:
#                     print(f"Insufficient variability in fitness values for {function_name} using {algorithm1} and {algorithm2}. Skipping t-test.")
#                     continue

#                 # Perform t-test
#                 t_stat, p_value = ttest_ind(fitness_values1, fitness_values2)
#                 print(f"\nStatistical Analysis for {function_name} using {algorithm1} and {algorithm2}:")
#                 print(f"P-value: {p_value}")

#                 # Define null hypothesis
#                 H0 = "There is no significant difference in the fitness values of the two algorithms."
#                 H1 = "There is a significant difference in the fitness values of the two algorithms."
                
#                 if p_value < 0.05:  # 95% confidence level
#                     print("Reject H0:", H1)
#                 else:
#                     print("Fail to reject H0:", H0)

#                 # Calculate means and standard deviations
#                 mean1, mean2 = np.mean(fitness_values1), np.mean(fitness_values2)
#                 std1, std2 = np.std(fitness_values1, ddof=1), np.std(fitness_values2, ddof=1)
                
#                 # Calculate confidence interval
#                 conf_interval = 1.96 * np.sqrt((std1**2 / len(fitness_values1)) + (std2**2 / len(fitness_values2)))
#                 mean_diff = mean1 - mean2
#                 ci_lower = mean_diff - conf_interval
#                 ci_upper = mean_diff + conf_interval

#                 print(f"95% Confidence Interval for the difference in means: ({ci_lower:.4f}, {ci_upper:.4f})")

#                 # Visualization
#                 x = np.linspace(-1, 1, 1000)
#                 y1 = norm.pdf(x, mean1, std1)
#                 y2 = norm.pdf(x, mean2, std2)

#                 plt.figure(figsize=(10, 6))
#                 plt.plot(x, y1, label=f'{algorithm1} (Mean: {mean1:.4f})', color='blue')
#                 plt.plot(x, y2, label=f'{algorithm2} (Mean: {mean2:.4f})', color='red')

#                 # Shade the confidence interval
#                 plt.fill_betweenx(y1, ci_lower, ci_upper, where=(x >= ci_lower) & (x <= ci_upper), color='lightblue', alpha=0.5, label='95% Confidence Interval')
                
#                 plt.title(f'Distribution of Fitness Values for {function_name}\n{algorithm1} vs {algorithm2}')
#                 plt.xlabel('Fitness Value')
#                 plt.ylabel('Probability Density')
#                 plt.legend()
#                 plt.grid()
                
#                  # Save the plot as PDF and PNG
#                 plt.savefig(f"{folder_name}/{function_name}_{algorithm1}_{algorithm2}_{suffix}.pdf")
#                 plt.savefig(f"{folder_name}/{function_name}_{algorithm1}_{algorithm2}_{suffix}.png")
                
#                 plt.show()
#     plt.close()

def statistical_analysis_with_visualization(results, suffix, folder_name):
    for i in range(len(results)):
        function_name = results[i]['function']
        algorithm_results = results[i]['results']
        
        # Create a dictionary to hold fitness values
        fitness_data = {}
        
        # Collect max_gbest_each_iter for each algorithm
        noise=1e-8 # small noise add to a force t-test
        for algorithm in algorithm_results:
            algorithm_name = algorithm['algorithm_name']
            # fitness_data[algorithm_name] = algorithm['max_gbest_each_iter']  # or use mean_gbest_each_iter
            fitness_data[algorithm_name] = algorithm['max_gbest_each_iter'] + noise * np.random.randn(len(algorithm['max_gbest_each_iter']))

        # Perform pairwise statistical analysis between algorithms
        algorithm_names = list(fitness_data.keys())
        for j in range(len(algorithm_names)):
            for k in range(j + 1, len(algorithm_names)):
                algorithm1 = algorithm_names[j]
                algorithm2 = algorithm_names[k]
                
                fitness_values1 = fitness_data[algorithm1]
                fitness_values2 = fitness_data[algorithm2]

                # Check for enough data points and variability
                if len(fitness_values1) < 2 or len(fitness_values2) < 2:
                    print(f"Not enough data to perform t-test for {function_name} using {algorithm1} and {algorithm2}.")
                    continue
                
                std1, std2 = np.std(fitness_values1, ddof=1), np.std(fitness_values2, ddof=1)
                # Check for low variability (close to identical values)
                if np.abs(np.mean(fitness_values1) - np.mean(fitness_values2)) < 1e-6 or std1 == 0 or std2 == 0:
                    print(f"Insufficient variability in fitness values for {function_name} using {algorithm1} and {algorithm2}. Skipping t-test.")
                    continue

                # Perform t-test
                t_stat, p_value = ttest_ind(fitness_values1, fitness_values2, equal_var=False)
                print(f"\nStatistical Analysis for {function_name} using {algorithm1} and {algorithm2}:")
                print(f"P-value: {p_value}")

                # Define null hypothesis
                H0 = "There is no significant difference in the fitness values of the two algorithms."
                H1 = "There is a significant difference in the fitness values of the two algorithms."
                
                if p_value < 0.05:  # 95% confidence level
                    print("Reject H0:", H1)
                else:
                    print("Fail to reject H0:", H0)

                # Calculate means and standard deviations
                mean1, mean2 = np.mean(fitness_values1), np.mean(fitness_values2)
                
                # Calculate confidence interval
                conf_interval = 1.96 * np.sqrt((std1**2 / len(fitness_values1)) + (std2**2 / len(fitness_values2)))
                mean_diff = mean1 - mean2
                ci_lower = mean_diff - conf_interval
                ci_upper = mean_diff + conf_interval

                print(f"95% Confidence Interval for the difference in means: ({ci_lower:.4f}, {ci_upper:.4f})")

                # Visualization
                x = np.linspace(min(mean1, mean2) - 3*max(std1, std2), max(mean1, mean2) + 3*max(std1, std2), 1000)
                y1 = norm.pdf(x, mean1, std1)
                y2 = norm.pdf(x, mean2, std2)

                plt.figure(figsize=(10, 6))
                plt.plot(x, y1, label=f'{algorithm1} (Mean: {mean1:.4f})', color='blue')
                plt.plot(x, y2, label=f'{algorithm2} (Mean: {mean2:.4f})', color='red')

                # Shade the confidence interval
                plt.fill_betweenx(y1, ci_lower, ci_upper, where=(x >= ci_lower) & (x <= ci_upper), color='lightblue', alpha=0.5, label='95% Confidence Interval')
                
                plt.title(f'Distribution of Fitness Values for {function_name}\n{algorithm1} vs {algorithm2}')
                plt.xlabel('Fitness Value')
                plt.ylabel('Probability Density')
                plt.legend()
                plt.grid()
                
                 # Save the plot as PDF and PNG
                plt.savefig(f"{folder_name}/{function_name}_{algorithm1}_{algorithm2}_{suffix}.pdf")
                plt.savefig(f"{folder_name}/{function_name}_{algorithm1}_{algorithm2}_{suffix}.png")
                
                plt.show()
    plt.close()

# Menyimpan hasil ke file
def save_results(results, folder_name):
    save_results_to_file(results, 'final', folder_name)
    save_chart_to_pdf(results, folder_name)

# Memproses log fitness
def process_fitness_log(results, alias_func_name):
    counter_alias = 0
    log_fitness = ''
    for func_result in results:
        temp_log_fitness = ''
        temp_val_fitness = 0
        temp_algo_name = ''
        for idx, result in enumerate(func_result['results']):
            if idx == 0:
                temp_val_fitness = result['best_fitness'].item()
                temp_algo_name = result['algorithm_name']
            else:
                # jika nilai fitness berupa nilai loss, maka gunakan comparasi if temp_val_fitness < result['best_fitness']:
                # tetapi jika nilai fitness berupa nilai yang profit atau invers dari loss, misal akurasi, gunakan > 
                if temp_val_fitness < result['best_fitness']:
                    temp_log_fitness = f"{alias_func_name[counter_alias]}-{temp_algo_name}-{temp_val_fitness:.2f}"
                else:
                    temp_val_fitness = result['best_fitness'].item()
                    temp_algo_name = result['algorithm_name']
                    temp_log_fitness = f"{alias_func_name[counter_alias]}-{temp_algo_name}-{temp_val_fitness:.2f}"
        log_fitness += ('--' if counter_alias else '') + temp_log_fitness
        counter_alias += 1
    return log_fitness

# Menyimpan hasil dalam folder
def create_save_folder(device, log_fitness, num_iterations, pop_size):
    info_param = f"{device}-{log_fitness}-{len(optimization_algorithms)}alg-it-{num_iterations}-ps-{pop_size}"
    timestamp = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y-%H-%M-%S')
    path = f"./log results/{info_param}-{timestamp}"
    os.makedirs(path, exist_ok=True)
    return path + '/'

# Menginisialisasi variabel dan menjalankan proses
log_fitness = process_fitness_log(results, alias_func_name)
save_folder = create_save_folder(my_device, log_fitness, num_iterations_all, pop_size_all)

# Tampilkan hasil dan simpan
# display_results_table(results)
display_results_table_df(results)
statistical_analysis_with_visualization(results, 'final_stat_viz', save_folder)
save_results(results, save_folder)

print("Done..!")