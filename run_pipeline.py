"""Reproduce the final rainfall–mobility analyses and report figures."""
# Resolve input files relative to the script and accept filesystem paths on the command line.
from pathlib import Path
import argparse  # Parse stage, directory and output options for a terminal run.
import json  # Print the analysis summary in the same format as the saved JSON.
import sys  # Control whether imports write compiled Python files into the extracted package.

# Keep imported analysis modules from leaving bytecode in the submission folder.
sys.dont_write_bytecode = True


def main():  # Choose the requested stage and produce its outputs.
    """Parse the command-line options and run raw preprocessing or the frozen-input report workflow."""
    # Use the module description as the summary shown by the command-line help.
    parser=argparse.ArgumentParser(description=__doc__)
    # Default to the data directory beside this script so it works from any working directory.
    parser.add_argument('--input-root',type=Path,default=Path(__file__).resolve().parent/'data',
                        # Describe the frozen-input directory in the help text.
                        help='Directory containing the supplied frozen input files.')
    # Write to an outputs directory under the caller's working directory unless a path is supplied.
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'),
                        # Describe the destination shared by the analysis products.
                        help='Destination for results, figures and tables.')
    # Accept the frozen-input analysis and raw preprocessing stages, with analysis as the default.
    parser.add_argument('--stage',choices=['analysis','preprocess'],default='analysis',
                        # Explain both stage choices in the command-line help.
                        help='Run the frozen-input analysis or optional raw-input preprocessing.')
    # Accept the path to the JSON list of preprocessing jobs.
    parser.add_argument('--raw-config',type=Path,help='JSON raw-input job list for --stage preprocess.')
    # Allow a numerical run to skip the report renderer.
    parser.add_argument('--skip-figures',action='store_true',help='Generate numeric analysis tables only.')
    args=parser.parse_args()  # Read and validate the options supplied for this invocation.
    # Compare absolute paths to catch two names for the same input and output directory.
    if args.input_root.resolve()==args.output_dir.resolve():
        # Stop before writing results alongside the source inputs.
        parser.error('Input and output directories must differ.')
    # Dispatch raw-data requests before importing the frozen-input analysis.
    if args.stage=='preprocess':
        # Require the job list that names the raw files and their processing options.
        if args.raw_config is None:parser.error('--raw-config is required for preprocessing.')
        # Load the optional preprocessing dependencies only on this branch.
        from src.raw_processing import prepare_raw
        # Execute the jobs described in the configuration using the selected directories.
        prepare_raw(args.raw_config,args.input_root,args.output_dir)
        return  # End the preprocessing run after writing its products.
    # Load the analysis driver and the common table readers and writers.
    from src.analysis import run,read,write
    # Calculate the event, peer, service and reference results from the frozen inputs.
    summary=run(args.input_root,args.output_dir)
    # Read the Pilot's observed pickup counts and historical reference medians.
    pilot=read(args.input_root/'pilot_hourly_baselines.csv')
    # Express each Pilot count as a percentage change from its reference median.
    pilot['pct_change']=100*(pilot.actual/pilot.baseline-1)
    # Require at least four reference observations and a baseline of ten pickups for the map.
    pilot['map_eligible']=pilot.reference_n.ge(4)&pilot.baseline.ge(10)
    # Mask unsupported percentage changes so they plot as missing.
    pilot.loc[~pilot.map_eligible,'pct_change']=float('nan')
    # Write the Pilot values and eligibility flags beside the other numerical results.
    write(args.output_dir/'results','pilot_snapshot.csv',pilot)
    if not args.skip_figures:  # Render the report products when the caller requested figures.
        # Import the plotting stack only when report rendering is needed.
        from src.figures import generate_report_outputs
        # Build the six figures and Table 1 from the freshly calculated results.
        generate_report_outputs(args.input_root,args.output_dir/'results',args.output_dir)
    # Show the same summary counts and recurring-zone names to the caller.
    print(json.dumps(summary,indent=2))


if __name__=='__main__':  # Run the command-line workflow only when this file is the entry point.
    main()  # Start the requested workflow with the current command-line arguments.
