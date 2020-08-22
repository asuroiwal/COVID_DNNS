import pandas as pd
from scripts.balanced_generator import BalancedDataGenerator

df_train = pd.read_csv('train_split.txt', sep=' ', index_col=None, header=0)
t = df_train.to_csv(header=None, index=False, sep=" ").strip('\n').split("\n")
trn_generator = BalancedDataGenerator(
        data_dir="os.path.join(args.data_dir, args.train_data_dir)",
        data_files=t,
        batch_size=8,
        mapping = None
    )
len(trn_generator)