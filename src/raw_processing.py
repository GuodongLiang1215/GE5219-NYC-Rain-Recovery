"""Optional TLC, MRMS and fixed-design baseline preparation from supplied raw files."""
from pathlib import Path  # Resolve local Parquet, GRIB and configuration file paths.
# Construct the timezone-aware UTC timestamp encoded in the MRMS header.
from datetime import datetime,timezone
# Decompress GRIB bytes, open the embedded image, read JSON and unpack binary scale metadata.
import gzip,io,json,struct
import numpy as np  # Transform rainfall grids, apply zone weights and construct baseline predictors.
import pandas as pd  # Align hourly series, handle time zones and write the preprocessing tables.
from PIL import Image  # Read the PNG-packed rainfall array embedded in each MRMS message.

# Interpret TLC civil timestamps and baseline clock features in New York local time.
TZ="America/New_York"
EPS=1e-6  # Allow for numerical precision at the fixed wet-hour and heavy-rain thresholds.


# Aggregate one source month and return its cleaning counts alongside the pickup totals.
def aggregate_trips(path,source,month,zone_ids):
    """Clean one TLC monthly Parquet and return UTC zone-hour completed pickups."""
    import duckdb  # Load DuckDB for querying TLC Parquet files directly.
    # Choose the timestamp column prefix used by this TLC source.
    prefix={"yellow":"tpep_","green":"lpep_","fhvhv":""}[source]
    # Construct the pickup and drop-off field names for the selected service.
    pickup,dropoff=prefix+"pickup_datetime",prefix+"dropoff_datetime"
    con=duckdb.connect()  # Open the DuckDB session used to inspect and aggregate the source file.
    con.execute("SET threads=2")  # Limit this monthly Parquet scan to two worker threads.
    # Limit the database session to one gigabyte of working memory.
    con.execute("SET memory_limit='1GB'")
    # Quote the file path as a SQL literal, doubling any embedded apostrophe.
    file="'"+str(Path(path)).replace("'","''")+"'"
    # Read the Parquet column names and types before constructing the cleaning query.
    schema={r[0]:r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({file})").fetchall()}
    # Require trip timestamps and pickup zone, plus a platform license for high-volume FHV records.
    required={pickup,dropoff,"PULocationID"}|({"hvfhs_license_num"} if source=="fhvhv" else set())
    # Report exactly which required source columns are missing before querying the data.
    if not required<=set(schema):raise ValueError(f"Missing TLC fields: {required-set(schema)}")
    # Require naive New York civil pickup times so the local-to-UTC conversion is applied once.
    if "WITH TIME ZONE" in schema[pickup]:raise ValueError("Expected NYC civil pickup timestamps")
    # Define the named calendar month's start and the exclusive start of the following month.
    begin=pd.Timestamp(month+'-01');end=(begin+pd.offsets.MonthBegin()).strftime('%Y-%m-%d')
    # Convert the permitted zone identifiers to integer literals for the SQL filter.
    ids=','.join(str(int(z)) for z in zone_ids)
    # Restrict high-volume FHV records to Uber and Lyft and allow all taxi rows through this test.
    platform="hvfhs_license_num IN ('HV0003','HV0005')" if source=='fhvhv' else 'TRUE'
    # Map the two retained FHV licenses to service names or use the taxi source name.
    service="CASE WHEN hvfhs_license_num='HV0003' THEN 'Uber' WHEN hvfhs_license_num='HV0005' THEN 'Lyft' ELSE 'Other' END" if source=='fhvhv' else f"'{source.title()}'"
    # Measure the trip duration in seconds for the validity check.
    duration=f"date_diff('second',{pickup},{dropoff})"
    # Bin pickup times by local clock hour before resolving daylight-saving ambiguity.
    query=(f"SELECT date_trunc('hour',{pickup}) AS hour_local_naive,\n"
           # Cast pickup zones to integers where possible and attach the service label.
           f"      TRY_CAST(PULocationID AS INTEGER) AS zone_id,{service} AS source_service,\n"
           # Give missing or out-of-month pickups the first exclusion status.
           f"      CASE WHEN {pickup} IS NULL OR {pickup}<TIMESTAMP '{month}-01' OR {pickup}>=TIMESTAMP '{end}' THEN 'outside_file_month_or_missing_pickup'\n"
           # Exclude trips with an unknown pickup zone or a zone outside the supplied NYC set.
           f"      WHEN PULocationID IS NULL OR PULocationID NOT IN ({ids}) THEN 'outside_nyc_zone'\n"
           # Exclude missing drop-offs, nonpositive durations and trips longer than six hours.
           f"      WHEN {dropoff} IS NULL OR {duration}<=0 OR {duration}>21600 THEN 'invalid_duration'\n"
           # Flag excluded or unreported platforms and mark trips passing all tests as retained.
           f"      WHEN NOT ({platform}) OR hvfhs_missing THEN 'other_platform' ELSE 'retained' END AS status\n"
           # Read the Parquet rows and expose a missing-license flag only for sources with that field.
           f"      FROM (SELECT *,{('hvfhs_license_num IS NULL' if source=='fhvhv' else 'FALSE')} AS hvfhs_missing FROM read_parquet({file}))")
    # Count trips by local hour, pickup zone, service and cleaning status in DuckDB.
    counts=con.execute(f'SELECT hour_local_naive,zone_id,source_service,status,COUNT(*)::BIGINT AS pickups FROM ({query}) GROUP BY ALL').df()
    con.close()  # Release the database session once the aggregated counts are in memory.
    # Keep retained counts for the mobility series and drop the classification column.
    q=counts[counts.status.eq('retained')].drop(columns='status').copy()
    # Localize civil hours, mark ambiguous or nonexistent times missing and convert valid hours to UTC.
    q['hour_utc']=pd.to_datetime(q.hour_local_naive).dt.tz_localize(TZ,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    # Count pickups lost because their local hour cannot be assigned uniquely to UTC.
    excluded_dst=int(q.loc[q.hour_utc.isna(),'pickups'].sum())
    # Aggregate the surviving counts by UTC hour, zone and service.
    q=q[q.hour_utc.notna()].groupby(['hour_utc','zone_id','source_service'],as_index=False).pickups.sum()
    # Record total source trips and the count passing the filters before timezone conversion.
    audit={'source_rows':int(counts.pickups.sum()),'retained_before_dst':int(counts.loc[counts.status.eq('retained'),'pickups'].sum()),
           # Add the daylight-saving exclusions and counts by mutually exclusive cleaning status.
           'ambiguous_or_nonexistent_pickups':excluded_dst,'status_counts':counts.groupby('status').pickups.sum().astype(int).to_dict()}
    return q,audit  # Return the UTC pickup table and its cleaning audit.


def sections(raw):  # Index the sections of a single GRIB2 message by section number.
    """Return the sections of a GRIB2 object used by MRMS Pass 2."""
    # Check the GRIB signature, edition number and end marker.
    assert raw[:4]==b'GRIB' and raw[7]==2 and raw[-4:]==b'7777'
    # Verify that the message length in the header matches the supplied bytes.
    assert int.from_bytes(raw[8:16],'big')==len(raw)
    parts={};i=16  # Start scanning after the 16-byte indicator section with an empty section index.
    while i<len(raw)-4:  # Read sections until the four-byte end marker is reached.
        # Decode this section's byte length and its numeric identifier.
        n=int.from_bytes(raw[i:i+4],'big');number=raw[i+4]
        # Require space for the section header and a unique section number.
        assert n>=5 and number not in parts
        # Store the full section bytes and advance to the next section boundary.
        parts[number]=raw[i:i+n];i+=n
    assert i==len(raw)-4  # Require the last section to end exactly where the terminator begins.
    return parts  # Return indexed sections for the MRMS decoder.


def signed(b):  # Interpret a big-endian GRIB sign-magnitude integer.
    """Decode the GRIB sign-magnitude scale factor."""
    n=int.from_bytes(b,'big');bits=len(b)*8  # Read the unsigned value and determine its total bit width.
    # Use the high bit as the sign and the remaining bits as the magnitude.
    return -(n&((1<<(bits-1))-1)) if n&(1<<(bits-1)) else n


def decode_mrms(path,crop_window):  # Decode the NYC portion of one hourly MRMS rainfall accumulation.
    """Read a local MRMS 1-hour Pass 2 GRIB.gz and return its NYC crop and start hour."""
    # Decompress the source file and split the GRIB2 message into sections.
    sec=sections(gzip.decompress(Path(path).read_bytes()))
    # Select identification, grid, product, representation, bitmap and data sections.
    s1,s3,s4,s5,s6,s7=[sec[k] for k in [1,3,4,5,6,7]]
    # Check the reference-time flag and require product definition template zero.
    assert s1[11]==3 and int.from_bytes(s4[7:9],'big')==0
    # Check that the product's category and parameter bytes match the expected MRMS field.
    assert s4[9:11]==bytes([6,37])
    # Require PNG packing template 41 and the flag for no separate bitmap.
    assert int.from_bytes(s5[9:11],'big')==41 and s6[5]==255
    # Verify the declared number of packed cells against the full MRMS grid dimensions.
    assert int.from_bytes(s5[5:9],'big')==7000*3500
    # Check the supported bit depth and PNG signature before decoding image data.
    assert s5[19] in [8,16] and s7[5:13]==b'\x89PNG\r\n\x1a\n'
    # Construct the UTC accumulation end time from the message's date and time fields.
    end=datetime(int.from_bytes(s1[12:14],'big'),*s1[14:19],tzinfo=timezone.utc)
    # Unpack the reference value and signed binary and decimal scale exponents.
    ref=struct.unpack('>f',s5[11:15])[0];e=signed(s5[15:17]);d=signed(s5[17:19])
    # Interpret the crop specification as its left column, top row, width and height.
    col,row,width,height=map(int,crop_window)
    # Open the PNG payload from the GRIB data section in memory.
    with Image.open(io.BytesIO(s7[5:])) as image:
        # Verify the raster dimensions against the expected 7,000 by 3,500 cells.
        assert image.size==(7000,3500)
        # Extract just the NYC window and use floating point for the scale conversion.
        pixels=np.array(image.crop((col,row,col+width,row+height)),dtype=np.float64)
    # Apply the GRIB reference and scaling factors to obtain rainfall in millimetres.
    crop=((ref+pixels*2.0**e)*10.0**(-d)).astype(np.float32)
    crop[(crop<0)|~np.isfinite(crop)]=np.nan  # Mask negative and nonfinite decoded rainfall values.
    # Return the crop, the start of the preceding accumulation hour and the grid definition.
    return crop,pd.Timestamp(end)-pd.Timedelta(hours=1),s3


# Aggregate each hourly rainfall crop to the supplied taxi-zone polygons.
def zone_rainfall(grib_files,weights_path,geometry_path):
    """Apply polygon-cell area weights, requiring at least 0.999999 valid area."""
    # Load the zone-by-cell area weights and their matching zone identifiers.
    with np.load(weights_path) as a:weights=a['weights'];zones=a['zone_ids']
    # Read the crop window defined for those weights.
    geometry=json.loads(Path(geometry_path).read_text(encoding='utf-8'))
    # Prepare hourly output rows and a reference for comparing grid definitions.
    rows=[];grid_definition=None
    for file in grib_files:  # Process each local hourly GRIB file in the supplied order.
        # Decode the NYC crop, its start hour and the source grid definition.
        crop,hour,definition=decode_mrms(file,geometry['crop_window'])
        # Save the first file's grid definition as the reference for later hours.
        if grid_definition is None:grid_definition=definition
        # Stop when a changed grid would invalidate the precomputed area weights.
        if grid_definition!=definition:raise ValueError('MRMS grid definition changed')
        # Flatten cells in weight order and calculate each zone's area fraction with valid rain.
        values=crop.ravel();coverage=weights@np.isfinite(values).astype(float)
        # Sum area-weighted rainfall, using zero for missing cells in this intermediate calculation.
        means=weights@np.nan_to_num(values,nan=0)
        # Mark rainfall unavailable below the required valid-area fraction of 0.999999.
        means[coverage<.999999]=np.nan
        for j,z in enumerate(zones):  # Read each zone's area-weighted result in the stored zone order.
            # Store the UTC hour, zone, mean rainfall and valid-area fraction together.
            rows.append(dict(hour_utc=hour.isoformat(),zone_id=int(z),rain_mean_mm=means[j],valid_area_fraction=coverage[j]))
    return pd.DataFrame(rows)  # Return one table row per input hour and zone.


# Identify wet-hour intervals separated by enough dry hours or by missing observations.
def local_episodes(values,min_dry_hours=3):
    """Segment wet hours using complete dry separation and explicit missing boundaries."""
    # Find finite readings above the 0.1 mm wet-hour threshold plus its tolerance.
    a=np.asarray(values);wet=np.flatnonzero(np.isfinite(a)&(a>.1+EPS))
    if not len(wet):return []  # Return an empty episode list when there are no qualifying wet hours.
    # Build cumulative missing-hour counts to test intervals between wet observations.
    missing=np.r_[0,np.cumsum(~np.isfinite(a))]
    # Detect any missing observations strictly between consecutive wet hours.
    unknown=missing[wet[1:]]-missing[wet[:-1]+1]>0
    # Split at missing intervals and at runs with at least the required number of known dry hours.
    cut=((np.diff(wet)>=min_dry_hours+1)&~unknown)|unknown
    # Translate split positions into start and exclusive end indices within the wet-hour list.
    first=np.r_[0,np.flatnonzero(cut)+1];last=np.r_[first[1:],len(wet)]
    # Return each episode's first wet hour and the index just after its last wet hour.
    return [(int(wet[s]),int(wet[e-1]+1)) for s,e in zip(first,last)]


# Build the local rain-event catalogue across the requested time interval.
def rainfall_events(hourly,start,end):
    """Return local rainfall segments on a complete UTC grid; retain weak episodes."""
    # Create a complete hourly UTC grid from the inclusive start to the exclusive end.
    grid=pd.date_range(start,end,tz='UTC',freq='h',inclusive='left')
    # Place each zone's rainfall on that grid so missing hours remain explicit.
    matrix=hourly.assign(hour_utc=pd.to_datetime(hourly.hour_utc,utc=True)).pivot(index='hour_utc',columns='zone_id',values='rain_mean_mm').reindex(grid)
    rows=[]  # Accumulate a catalogue row for each zone and segmented episode.
    for z in matrix:  # Segment the hourly series for each zone separately.
        a=matrix[z].to_numpy()  # Read the zone's ordered rainfall values as an array.
        # Iterate over this zone's wet intervals with exclusive right boundaries.
        for left,right in local_episodes(a):
            # Select all hourly amounts within the episode, including any intervening dry hours.
            v=a[left:right]
            # Store the zone and its UTC start and exclusive end timestamps.
            rows.append(dict(zone_id=int(z),start_utc=grid[left].isoformat(),end_utc=(grid[right-1]+pd.Timedelta(hours=1)).isoformat(),
                # Record duration, total and peak rainfall, and whether the peak reaches the heavy-rain threshold.
                duration_h=right-left,total_mm=float(v.sum()),peak_mm=float(v.max()),strong=bool(v.max()>=9.4-EPS),
                # Flag whether every rainfall value in the episode is finite.
                mean_rain_complete=bool(np.isfinite(v).all())))
    # Return the event records, including episodes below the heavy-rain threshold.
    return pd.DataFrame(rows)


def features(grid):  # Build the fixed design matrix used for expected hourly pickups.
    """Construct the fixed 173-column hour-of-week, trend and seasonal design."""
    # Derive New York civil timestamps and the 168 possible local weekday-hour slots.
    local=grid.tz_convert(TZ);civil=local.tz_localize(None);slot=np.asarray(local.dayofweek*24+local.hour)
    # Express the fractional local day of year as an angle for annual harmonics.
    angle=2*np.pi*np.asarray(local.dayofyear-1+local.hour/24)/365.25
    # Measure the linear time trend in years relative to 1 January 2023.
    trend=(civil-pd.Timestamp('2023-01-01')).total_seconds().to_numpy()/86400/365.25
    # Combine 168 clock indicators, one trend and annual and semiannual sine/cosine terms.
    return np.column_stack([np.eye(168)[slot],trend,np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])


# Fit expected pickups on eligible normal hours and predict at the requested indices.
def fit_baseline(hour_utc,actual,objective_normal,excluded,requested_indices):
    """Fit the original Ridge baseline using supplied fixed normal/exclusion masks."""
    # Load the Ridge estimator only for jobs that actually fit a baseline.
    from sklearn.linear_model import Ridge
    # Convert all timestamps to UTC and generate the corresponding fixed design rows.
    grid=pd.DatetimeIndex(pd.to_datetime(hour_utc,utc=True));X=features(grid)
    # Restrict training to eligible, unexcluded hours with finite, nonnegative actual pickups.
    actual=np.asarray(actual,float);mask=np.asarray(objective_normal,bool)&~np.asarray(excluded,bool)&np.isfinite(actual)&(actual>=0)
    # Require at least 1,000 training hours for this fit.
    if mask.sum()<1000:raise ValueError('At least 1,000 eligible training hours are required')
    # Identify the local weekday and hour of every training and prediction row.
    slots=np.asarray(grid.tz_convert(TZ).weekday*24+grid.tz_convert(TZ).hour)
    # Count eligible training observations in all 168 local clock cells.
    support=np.bincount(slots[mask],minlength=168)
    # Interpret the requested positions as integer indices into the input grid.
    ix=np.asarray(requested_indices,int)
    # Check whether any requested prediction has no training hour in its local clock cell.
    if np.any(support[slots[ix]]==0):
        # Reject predictions that would extrapolate to an unsupported weekday and hour.
        raise ValueError('Training has no support for a requested local weekday/hour cell')
    # Fit ridge regression to log-transformed counts with the original penalty and solver.
    model=Ridge(alpha=1,fit_intercept=True,solver='cholesky').fit(X[mask],np.log1p(actual[mask]))
    # Invert the log transformation and constrain pickup predictions to be nonnegative.
    prediction=np.maximum(np.expm1(model.predict(X[ix])),0)
    return prediction  # Return predictions in the order of the requested indices.


# Estimate clock-specific reference gaps while weighting normal dates equally.
def centered_reference(raw_gaps,dates,clock_cells,requested_clock_cells):
    """Return equal-date mean normal residual ratios for matching local clock classes."""
    # Associate each raw gap with its date and local clock class.
    q=pd.DataFrame({'date':dates,'clock':clock_cells,'gap':raw_gaps})
    # Average within clock and date first, then weight the resulting date means equally.
    means=q.groupby(['clock','date']).gap.mean().groupby('clock').mean()
    # Return the centers in the requested clock order, leaving absent classes missing.
    return means.reindex(requested_clock_cells).to_numpy()


# Dispatch the preprocessing jobs specified in a local JSON configuration.
def prepare_raw(config_path,input_root,output_dir):
    """Execute the explicitly listed local preprocessing jobs; no downloads are performed."""
    # Resolve the input and output paths and create the destination directory.
    root=Path(input_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    # Read the job configuration and then the zone identifiers used by its trip filters.
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'));zone_ids=pd.read_csv(root/cfg['zone_labels']).zone_id
    audits=[]  # Collect per-file trip cleaning summaries for the final JSON.
    for job in cfg.get('trip_jobs',[]):  # Run the listed trip-month jobs, if any.
        # Clean and aggregate this source-month using the configured zone set.
        frame,audit=aggregate_trips(root/job['file'],job['source'],job['month'],zone_ids)
        # Name the hourly pickup file by source and month.
        name=f"{job['source']}_{job['month']}_hourly.csv.gz"
        # Write compressed counts without row indices and with a fixed gzip timestamp.
        frame.to_csv(out/name,index=False,compression={'method':'gzip','mtime':0})
        # Associate the cleaning counts with the source and output filenames.
        audits.append(dict(file=job['file'],output=name,**audit))
    rain=cfg.get('rainfall')  # Look up the optional rainfall job in the configuration.
    if rain:  # Process rainfall only when that job is present and populated.
        # Resolve each configured hourly GRIB file relative to the input root.
        paths=[root/name for name in rain['files']]
        # Apply the supplied polygon weights and grid metadata to the listed hours.
        frame=zone_rainfall(paths,root/rain['weights'],root/rain['geometry'])
        # Save hourly rainfall and area coverage as a reproducible gzip CSV.
        frame.to_csv(out/'zone_hourly_rainfall.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        # Segment events only when both study-period boundaries are specified.
        if 'study_start' in rain and 'study_end_exclusive' in rain:
            # Write the local rainfall catalogue for the configured time interval.
            rainfall_events(frame,rain['study_start'],rain['study_end_exclusive']).to_csv(out/'local_rainfall_events.csv',index=False)
    # Process each optional fixed-design baseline fitting request.
    for job in cfg.get('baseline_jobs',[]):
        # Open the request archive for its observations, masks and prediction indices.
        with np.load(root/job['file']) as a:
            # Fit the original baseline with this request's training masks and requested rows.
            prediction=fit_baseline(a['hour_utc'],a['actual'],a['objective_normal'],a['excluded'],a['requested_indices'])
            # Export each requested UTC timestamp beside its predicted count.
            pd.DataFrame({'hour_utc':a['hour_utc'][a['requested_indices']],'prediction':prediction}).to_csv(out/job['output'],index=False)
    # Write all recorded trip-cleaning summaries to the preprocessing output directory.
    (out/'preprocessing_summary.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
