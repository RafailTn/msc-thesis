import argparse
import subprocess
import os
import requests
from autogluon.tabular import TabularPredictor

# The official UCSC URL for hg38 470-way conservation
url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw"
local_filename = "hg38.phastCons470way.bw"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-target_fasta', required=True, help='FASTA file containing target/mRNA sequences')
    parser.add_argument('-query_fasta', required=True, help='FASTA file containing query/miRNA sequences')
    parser.add_argument('-conservation_path', help='file containing conservation scores. If not provided and there is no bw, phastcons470way is downloaded.')
    parser.add_argument('-coords', help='file with the hg38 coordinates of the target sequences in tsv format')
    parser.add_argument('-keep_files', action='store_true', help='Keep intermediate files')
    parser.add_argument('-o', type=str, default='./results.tsv', help='Output file')
    parser.add_argument('-threshold', type=float, default=0.5, help='Probability threshold above which predictions are condidered interactions')
    parser.add_argument('-threads', type=int, default=4, help='Number of threads to use (ie for intarna)')
    args = parser.parse_args()

    if args.threads < 1:
        print("Error: Thread count must be at least 1", file=sys.stderr)
        return 1
    
    # Use CPU count as reasonable upper bound suggestion
    cpu_count = os.cpu_count() or 4
    if args.threads > cpu_count * 2:
        print(f"Warning: Using {args.threads} threads on a system with {cpu_count} CPUs. "
              f"Consider using {cpu_count} to {cpu_count * 2} threads for optimal performance.",
              file=sys.stderr)
    

    # Optional: Download the file if it doesn't exist
    if not os.path.exists(local_filename) and not args.conservation_path:
        print("Downloading bigWig file...")
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk) 
        args.conservation_path = local_filename

    elif os.path.exists(local_filename) and not args.conservation_path:
        args.conservation_path = local_filename
    

    os.makedirs('./temp/', exists_ok=True) 
    intarna_cmd = f'python3 intarna_parallel.py {args.target_fasta} {args.query_fasta} -o ./temp/intarna_results.tsv'
    ensemble_cmd = f'python3 intarna_parallel.py {args.target_fasta} {args.query_fasta} -o ./temp/intarna_results_ensemble.tsv --ensemble''
    intarna_result = subprocess.run(intarna_cmd, timeout=600) 
    ensemble_result = subprocess.run(ensemble_cmd, timeout=600)

    merge_cmd = f'python3 merge_intarna.py -m ./temp/intarna_results.tsv -e ./temp/intarna_results_ensemble.tsv -o intarna_merged.tsv'
    merge_result = subprocess.run(merge_cmd, timeout=600)

    best_structs_cmd = f'python3 best_intarna.py --intarna ./temp/intarna_merged.tsv --mre-fasta {args.target_fasta} --mirna-fasta {args.query_fasta} --output ./temp/intarna_best.tsv'
    best_result = subprocess.run(best_structs_cmd, timeout=600)

    feature_extraction_cmd = f'python3 intarna_fe.py --intarna ./temp/intarna_best.tsv --mre-fasta {args.target_fasta} --mirna-fasta {args.query_fasta} --conservation_path {args.coords} --output ./temp/samples_4_pred_noseq.csv'
    feature_extraction_result = subprocess.run(feature_extraction_cmd, timeout=600)
    
    seq_ft_extraction_cmd = f'python3 seq_features_ml.py ./temp/samples_4_pred_noseq.csv -o ./temp/samples_4_pred.csv'
    seq_ft_extraction_result = subprocess.run(seq_ft_extraction_cmd, timeout=600)
    
    predictor = TabularPredictor.load()
    input4pred = pl.read_csv('./temp/samples_4_pred.csv')
    input4pred = input4pred.to_pandas()
    predictions = predictor.predict(input4pred, args.threshold) 
    input_seqs = pl.from_pandas(input4pred).select(['mre_sequence', 'mirna_sequence'])
    output = input_seqs.with_columns(predictions=predictions) 
    output.write_csv(args.o, separator='/t')

    if not args.keep_files:
        current_path = os.getcwd()
        if 'msc-thesis' in current_path:
            rm_cmd = 'rm -rf ./temp'
            rm_res = subprocess.run(rm_cmd, timeout=600)

if __name__ == "__main__":
    main()
