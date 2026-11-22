import pandas as pd

a = pd.read_csv("results_llama3.2_3b_prompt_only_t1.0.csv")
b = pd.read_csv("results_llama3.2_3b_schema_t1.0.csv")
for i in [4, 49]:
    print(f"idx {i} | gerçek: {a.loc[i, 'true']}")
    print("  prompt_only:", a.loc[i, "pred"], "|", a.loc[i, "raw"])
    print("  schema     :", b.loc[i, "pred"], "|", b.loc[i, "raw"])