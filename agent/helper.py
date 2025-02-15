import os

def get_agent_number(runs_folder_path):
    if not os.path.exists(runs_folder_path) or not os.listdir(runs_folder_path):
        return 1
    
    run_folders = os.listdir(runs_folder_path)
    run_numbers = [int(folder_name) for folder_name in run_folders if folder_name.isdigit()]
    next_run_number = max(run_numbers) + 1
    return next_run_number

def changement_calculator(start_price, end_price):
    percentage_change = ((end_price - start_price) / start_price) * 100
    return percentage_change
