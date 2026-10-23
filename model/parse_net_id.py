import sys
import csv

def get_network_architecture(network_id, csv_file_path='sampled_1k_15ms_a57.csv'):
    """
    Parse the CSV file and return the network architecture for a given network_id.
    
    Args:
        network_id: The ID of the network to retrieve
        csv_file_path: Path to the CSV file (default: 'sampled_1k_15ms_a57.csv')
    
    Returns:
        A nested array containing [stage_0_cfg, stage_1_cfg, stage_2_cfg, 
        stage_3_cfg, stage_4_cfg, stage_5_cfg, stage_6_cfg], where each 
        stage_cfg is a list of integers parsed from the underscore-separated string.
    """
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['network_id']) == network_id:
                architecture = []
                for stage_idx in range(7):  # stages 0-6
                    stage_key = f'stage_{stage_idx}_cfg'
                    stage_cfg_str = row[stage_key]
                    # Parse the underscore-separated values into integers
                    stage_cfg = [int(x) for x in stage_cfg_str.split('_')]
                    architecture.append(stage_cfg)
                return architecture
    
    raise ValueError(f"Network ID {network_id} not found in the CSV file")


# _id = int(sys.argv[1])
# print(get_network_architecture(_id))