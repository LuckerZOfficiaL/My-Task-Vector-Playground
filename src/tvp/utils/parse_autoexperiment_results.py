import json

def parse_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by the divider
    sections = content.strip().split("\n\n---\n\n")

    results = []

    for section in sections:
        if "\nResults ATM and TA:" in section:
            section = section.split("\nResults ATM and TA: ")[1]
        if "Results ATM and TA:" in section:
            section = section.split("Results ATM and TA: ")[1]
        if "\n\n---" in section:
            section = section.split("\n\n---")[0]

        section = section.replace("'", '"')
        results.append(section)



    return results


def convert_to_dict(section):
    """
    Converts a section string to a Python dictionary if possible.
    """
    try:
        parsed_section = json.loads(section)
        return parsed_section
    except json.JSONDecodeError as e:
        print(f"Failed to parse section as JSON: {e}")
        return section


if __name__ == "__main__":
    
    input_file = "autoexperiment_output_log_2024_12_04_at_09_31_00__filtered.log"
    parsed_results = parse_file(input_file)

    avg_acc_normalized_atm_11_tvs = []
    avg_acc_atm_11_tvs = []
    avg_acc_normalized_ta_11_tvs = []
    avg_acc_ta_11_tvs = []
    cos_sims_11_tvs = []

    # Convert and print results
    for i, result in enumerate(parsed_results, 1):
        print(f"\n\n--- Result {i} ---\n\n")

        parsed_dict = convert_to_dict(result)

        avg_acc_normalized_atm_11_tvs.append(
            parsed_dict["avg_acc_normalized_atm"]
        )

        avg_acc_atm_11_tvs.append(
            parsed_dict["avg_acc_atm"]
        )

        avg_acc_normalized_ta_11_tvs.append(
            parsed_dict["avg_acc_normalized_ta"]
        )

        avg_acc_ta_11_tvs.append(
            parsed_dict["avg_acc_ta"]
        )

        cos_sims_11_tvs.append(
            parsed_dict["cos_sim_atm_ta"]
        )

    print(f"avg_acc_normalized_atm_11_tvs = {avg_acc_normalized_atm_11_tvs}")
    print(f"\n\n")

    print(f"avg_acc_atm_11_tvs = {avg_acc_atm_11_tvs}")
    print(f"\n\n")

    print(f"avg_acc_normalized_ta_11_tvs = {avg_acc_normalized_ta_11_tvs}")
    print(f"\n\n")

    print(f"avg_acc_ta_11_tvs = {avg_acc_ta_11_tvs}")
    print(f"\n\n")

    print(f"cos_sims_11_tvs = {cos_sims_11_tvs}")
    print(f"\n\n")


