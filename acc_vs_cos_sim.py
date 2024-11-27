import ast

# Initialize dictionaries to store values for each specified key, indexed by cardinality
applied_tvs_by_cardinality = {5: [], 6: [], 7: []}
cos_sims_by_cardinality = {5: [], 6: [], 7: []}
avg_norm_atm_accs_by_cardinality = {5: [], 6: [], 7: []}
avg_atm_accs_by_cardinality = {5: [], 6: [], 7: []}
avg_norm_ta_accs_by_cardinality = {5: [], 6: [], 7: []}
avg_ta_accs_by_cardinality = {5: [], 6: [], 7: []}

# Define a mapping of keys to the respective dictionaries by cardinality
key_to_dict = {
    'applied_task_vectors': applied_tvs_by_cardinality,
    'cos_sim_atm_ta': cos_sims_by_cardinality,
    'avg_acc_normalized_atm': avg_norm_atm_accs_by_cardinality,
    'avg_acc_atm': avg_atm_accs_by_cardinality,
    'avg_acc_normalized_ta': avg_norm_ta_accs_by_cardinality,
    'avg_acc_ta': avg_ta_accs_by_cardinality
}

# Read the file and process each line
with open("autoexperiment_output_log_2024_11_08_at_12_19_00_compressed.log", "r") as file:
    for line in file:
        try:
            # Convert the string representation of the dictionary to an actual dictionary
            data_dict = ast.literal_eval(line.strip())
            
            # Check the length of 'applied_task_vectors' and proceed only for cardinalities 5, 6, or 7
            cardinality = len(data_dict['applied_task_vectors'])
            if cardinality in [5, 6, 7]:
                # Extract values for specified keys and append them to the respective lists in dictionaries
                for key, dict_by_cardinality in key_to_dict.items():
                    if key in data_dict:
                        dict_by_cardinality[cardinality].append(data_dict[key])

        except (ValueError, SyntaxError) as e:
            print(f"Error processing line: {line}")
            print(f"Error details: {e}")

# Now print each dictionary as a Python assignment statement
for cardinality in [5, 6, 7]:
    print(f"applied_tvs_{cardinality}_tvs = {applied_tvs_by_cardinality[cardinality]}\n")
    print(f"cos_sims_{cardinality}_tvs = {cos_sims_by_cardinality[cardinality]}\n")
    print(f"avg_norm_atm_accs_{cardinality}_tvs = {avg_norm_atm_accs_by_cardinality[cardinality]}\n")
    print(f"avg_atm_accs_{cardinality}_tvs = {avg_atm_accs_by_cardinality[cardinality]}\n")
    print(f"avg_norm_ta_accs_{cardinality}_tvs = {avg_norm_ta_accs_by_cardinality[cardinality]}\n")
    print(f"avg_ta_accs_{cardinality}_tvs = {avg_ta_accs_by_cardinality[cardinality]}\n")

    print(f"\n\n\n")
    
    
    
    # print(f"len(zip(applied_tvs_{cardinality}_tvs, cos_sims_{cardinality}_tvs, avg_norm_atm_accs_{cardinality}_tvs, "
    #       f"avg_atm_accs_{cardinality}_tvs, avg_norm_ta_accs_{cardinality}_tvs, avg_ta_accs_{cardinality}_tvs)) = "
    #       f"{len(list(zip(applied_tvs_by_cardinality[cardinality], cos_sims_by_cardinality[cardinality], "
    #       f"avg_norm_atm_accs_by_cardinality[cardinality], avg_atm_accs_by_cardinality[cardinality], "
    #       f"avg_norm_ta_accs_by_cardinality[cardinality], avg_ta_accs_by_cardinality[cardinality])))}\n")
