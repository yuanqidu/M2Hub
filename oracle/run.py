import argparse
from oracle import Oracle

def rf_scm_magpie(oracle, test_data):
    predictions = oracle.predict([test_data])
    return predictions

def substrate_matching(oracle, test_data):
    matches = oracle.substrate_match(test_data)
    return matches

def main(task_name, test_data, oracle_to_run):
    # Initialize the Oracle
    oracle = Oracle(task_name=task_name)

    if oracle_to_run == 'rf_scm_magpie':
        return rf_scm_magpie(oracle, test_data)
    elif oracle_to_run == 'substrate_matching':
        return substrate_matching(oracle, test_data)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional arguments
    parser.add_argument("-t", "--Task", help = "Name of the Task")
    parser.add_argument("-d", "--Data", help = "Test Data")
    parser.add_argument("-o", "--Oracle", choices = ['rf_scm_magpie', 'substrate_matching'], help = "Which oracle to run")

    # Read arguments from command line
    args = parser.parse_args()

    main(task_name=args.Task, test_data=args.Data, oracle_to_run=args.Oracle)
