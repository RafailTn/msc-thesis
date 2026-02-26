import os
import subprocess
from Bio import SeqIO
import multiprocessing 
from typing import List
import pandas as pd
import numpy as np
import sys

def write_fasta(col:str, path:str = './data/', output_path:str='./data/'):
    dfs = [pd.read_csv(os.path.join(path,df), sep='\t') for df in os.listdir(path) if df.endswith('.tsv')]
    filenames = [fn for fn in os.listdir(path) if fn.endswith('.tsv')]
    for df, fn in zip(dfs,filenames):
        with open(os.path.join(path, f"{fn.replace('.tsv', f'_{col}.fasta')}"), 'w') as infile:
            for i, seq in enumerate(df[col]):
                infile.write(f">seq{i}\n{seq}\n")

def make_chimeric(df: pd.DataFrame):
    miRNA = df[1]
    MRE = df[0]
    chim = MRE + miRNA
    return chim

def run_rnaduplex_on_pairs(path:str=os.getcwd(), output_path='./data/'):
    """
    Parses two multi-fasta files, runs RNAduplex on each pair,
    and writes the results to an output file.

    Args:
        path: Path to the input files
    """
    gene_fastas = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('gene.fasta')]
    mirnas = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('noncodingRNA.fasta')] 
    for g in gene_fastas:
        for m in mirnas:
            if g.split('/')[-1].split('.')[0].replace('_gene', '') == m.split('/')[-1].split('.')[0].replace('_noncodingRNA', ''):
                # Open the output file before starting the loop
                with open(os.path.join(path, f'{g.split('/')[-1].split('.')[0].replace('_gene', '')}_duplex.txt'), 'w') as out_file:
                    records1 = SeqIO.parse(g, "fasta")
                    records2 = SeqIO.parse(m, "fasta")

                    # Loop through both files simultaneously using zip
                    for rec1, rec2 in zip(records1, records2):
                        sequence_pair_input = f">{rec1.id}\n{rec1.seq}\n>{rec2.id}\n{rec2.seq}\n"
                        command = ["RNAduplex", "--noLP", "-e 15"]

                        try:
                            result = subprocess.run(
                                command,
                                input=sequence_pair_input,
                                capture_output=True,
                                text=True,
                                check=True
                            )

                            # Write a header and the RNAduplex output to the file
                            out_file.write(f"Interaction for: {rec1.id} & {rec2.id}\n")
                            out_file.write(result.stdout) # result.stdout already has a newline
                            out_file.write("-" * 30 + "\n")

                        except FileNotFoundError:
                            print("Error: 'RNAduplex' command not found.", file=sys.stderr)
                            print("Please ensure ViennaRNA is installed and in your system's PATH.", file=sys.stderr)
                            sys.exit(1)
                        except subprocess.CalledProcessError as e:
                            print(f"Error processing pair {rec1.id} & {rec2.id}: {e.stderr}", file=sys.stderr)

    print("Success! All results have been written")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df_path', help="If sequences are in a tsv, convert them to fastas")
    parser.add_argument('-cols', nargs='+', type=str, help='Column from df to make fasta')
    parser.add_argument('-fasta_path', required=True, help='Paths for fasta files')
    parser.parse_argument('-output_path', help='Output path')
    args = parser.parse_args()

    if args.df_path and args.cols:
        for col in args.cols:
            write_fasta(col=col, path=args.df_path, output_path=args.output_path)
        run_rnaduplex_on_pairs(path=args.fasta_path, output_path=args.output_path)  
   
    else:
        run_rnaduplex_on_pairs(path=args.fasta_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
