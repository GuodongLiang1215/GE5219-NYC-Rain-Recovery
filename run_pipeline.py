"""Reproduce the final rainfall–mobility analyses and report figures."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path(__file__).resolve().parent/'data',
                        help='Directory containing the supplied frozen input files.')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'),
                        help='Destination for results, figures and tables.')
    parser.add_argument('--stage',choices=['analysis','preprocess'],default='analysis',
                        help='Run the frozen-input analysis or optional raw-input preprocessing.')
    parser.add_argument('--raw-config',type=Path,help='JSON raw-input job list for --stage preprocess.')
    parser.add_argument('--skip-figures',action='store_true',help='Generate numeric analysis tables only.')
    args=parser.parse_args()
    if args.input_root.resolve()==args.output_dir.resolve():
        parser.error('Input and output directories must differ.')
    if args.stage=='preprocess':
        if args.raw_config is None:parser.error('--raw-config is required for preprocessing.')
        from src.raw_processing import prepare_raw
        prepare_raw(args.raw_config,args.input_root,args.output_dir)
        return
    from src.analysis import run,read,write
    summary=run(args.input_root,args.output_dir)
    pilot=read(args.input_root/'pilot_hourly_baselines.csv')
    pilot['pct_change']=100*(pilot.actual/pilot.baseline-1)
    pilot['map_eligible']=pilot.reference_n.ge(4)&pilot.baseline.ge(10)
    pilot.loc[~pilot.map_eligible,'pct_change']=float('nan')
    write(args.output_dir/'results','pilot_snapshot.csv',pilot)
    if not args.skip_figures:
        from src.figures import generate_report_outputs
        generate_report_outputs(args.input_root,args.output_dir/'results',args.output_dir)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
