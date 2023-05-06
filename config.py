import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--test_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

parser.add_argument("--plm", default=None, type=str, required=True,
                        help="pre-trained model")

parser.add_argument("--mytype", default=None, type=str, required=True,
                        help="pre-trained model")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--eval_test_file_name", default="eval.txt", type=str)
parser.add_argument("--datasets", default="lap14", type=str)

parser.add_argument("--max_seq_length", default=100, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
parser.add_argument("--batch_size", default=10, type=int, help="Total batch size for training.")
parser.add_argument("--epochs", default=12, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--lr", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--eval_steps", default=50, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--fp16", action='store_true', help="Whether to use apex.")
args = parser.parse_args()