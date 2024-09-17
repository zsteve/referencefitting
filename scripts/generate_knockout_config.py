import numpy as np
import pandas as pd
import yaml
import json
import itertools
import copy
import os

# config_path = "tools/BoolODE/config-files/trifurcating.yaml"
# config_path = "../../tools/BoolODE/config-files/beeline-inputs-synthetic.yaml"
# config_path = "../../tools/BoolODE/config-files/beeline-inputs-boolean.yaml"
config_path = "../../tools/BoolODE/config-files/ventre.yaml"
with open(config_path) as stream:
    config = yaml.safe_load(stream)

config_new = copy.deepcopy(config)
    
for job in config['jobs']:
    print(f"job : {job['name']}")
    rules_file = job['model_definition']
    ic_file = job['model_initial_conditions']

    base_path = os.path.dirname(os.path.dirname(config_path))
    genes = pd.read_csv(os.path.join(base_path, "data", rules_file), sep = "\t").Gene
    df_rules = pd.read_csv(os.path.join(base_path, "data", rules_file), sep = "\t")
    df_ic = pd.read_csv(os.path.join(base_path, "data", ic_file), sep = "\t")
    # don't knock-out IC genes
    ic_genes = list(itertools.chain(*[json.loads(str(x).replace("'", "\"")) for x in df_ic.Genes]))

    for gene in genes:
        if gene not in ic_genes:
            gene_idx = np.where(df_rules.Gene == gene)[0][0]
            df_rules_ko = df_rules.copy()
            df_ic_ko = df_ic.copy()
            df_rules_ko.iloc[gene_idx, :].Rule = gene
            df_ic_ko.loc[df_ic_ko.shape[0], :] = {"Genes" : f"['{gene}']", "Values" : "[0]"}
            df_rules_ko.to_csv(os.path.join(base_path, "data", f"{os.path.splitext(rules_file)[0]}_ko_{gene}.txt"), sep = "\t", index = False)
            df_ic_ko.to_csv(os.path.join(base_path, "data", f"{os.path.splitext(ic_file)[0].split('_ics')[0]}_ko_{gene}_ics.txt"), sep = "\t", index = False)
            job_new = job.copy()
            job_new['name'] = f"{job_new['name']}_ko_{gene}"
            job_new['model_definition'] = f"{os.path.splitext(rules_file)[0]}_ko_{gene}.txt"
            job_new['model_initial_conditions'] = f"{os.path.splitext(ic_file)[0].split('_ics')[0]}_ko_{gene}_ics.txt"
            config_new['jobs'].append(job_new)

with open(os.path.join(os.path.dirname(config_path), f"{os.path.splitext(os.path.basename(config_path))[0]}_ko.yml"), 'w') as outfile:
    yaml.dump(config_new, outfile, default_flow_style=False)
        
