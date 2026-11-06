"""Optional TLC, MRMS and fixed-design baseline preparation from supplied raw files."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,io,json,struct
import numpy as np
import pandas as pd
from PIL import Image

TZ="America/New_York"
EPS=1e-6


def aggregate_trips(path,source,month,zone_ids):
    """Clean one TLC monthly Parquet and return UTC zone-hour completed pickups."""
    import duckdb
    prefix={"yellow":"tpep_","green":"lpep_","fhvhv":""}[source]
    pickup,dropoff=prefix+"pickup_datetime",prefix+"dropoff_datetime"
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")
    file="'"+str(Path(path)).replace("'","''")+"'"
    schema={r[0]:r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({file})").fetchall()}
    required={pickup,dropoff,"PULocationID"}|({"hvfhs_license_num"} if source=="fhvhv" else set())
    if not required<=set(schema):raise ValueError(f"Missing TLC fields: {required-set(schema)}")
    if "WITH TIME ZONE" in schema[pickup]:raise ValueError("Expected NYC civil pickup timestamps")
    begin=pd.Timestamp(month+'-01');end=(begin+pd.offsets.MonthBegin()).strftime('%Y-%m-%d')
    ids=','.join(str(int(z)) for z in zone_ids)
    platform="hvfhs_license_num IN ('HV0003','HV0005')" if source=='fhvhv' else 'TRUE'
    service="CASE WHEN hvfhs_license_num='HV0003' THEN 'Uber' WHEN hvfhs_license_num='HV0005' THEN 'Lyft' ELSE 'Other' END" if source=='fhvhv' else f"'{source.title()}'"
    duration=f"date_diff('second',{pickup},{dropoff})"
    query=f"""SELECT date_trunc('hour',{pickup}) AS hour_local_naive,
      TRY_CAST(PULocationID AS INTEGER) AS zone_id,{service} AS source_service,
      CASE WHEN {pickup} IS NULL OR {pickup}<TIMESTAMP '{month}-01' OR {pickup}>=TIMESTAMP '{end}' THEN 'outside_file_month_or_missing_pickup'
      WHEN PULocationID IS NULL OR PULocationID NOT IN ({ids}) THEN 'outside_nyc_zone'
      WHEN {dropoff} IS NULL OR {duration}<=0 OR {duration}>21600 THEN 'invalid_duration'
      WHEN NOT ({platform}) OR hvfhs_missing THEN 'other_platform' ELSE 'retained' END AS status
      FROM (SELECT *,{('hvfhs_license_num IS NULL' if source=='fhvhv' else 'FALSE')} AS hvfhs_missing FROM read_parquet({file}))"""
    counts=con.execute(f'SELECT hour_local_naive,zone_id,source_service,status,COUNT(*)::BIGINT AS pickups FROM ({query}) GROUP BY ALL').df()
    con.close()
    q=counts[counts.status.eq('retained')].drop(columns='status').copy()
    q['hour_utc']=pd.to_datetime(q.hour_local_naive).dt.tz_localize(TZ,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    excluded_dst=int(q.loc[q.hour_utc.isna(),'pickups'].sum())
    q=q[q.hour_utc.notna()].groupby(['hour_utc','zone_id','source_service'],as_index=False).pickups.sum()
    audit={'source_rows':int(counts.pickups.sum()),'retained_before_dst':int(counts.loc[counts.status.eq('retained'),'pickups'].sum()),
           'ambiguous_or_nonexistent_pickups':excluded_dst,'status_counts':counts.groupby('status').pickups.sum().astype(int).to_dict()}
    return q,audit


def sections(raw):
    """Return the sections of a GRIB2 object used by MRMS Pass 2."""
    assert raw[:4]==b'GRIB' and raw[7]==2 and raw[-4:]==b'7777'
    assert int.from_bytes(raw[8:16],'big')==len(raw)
    parts={};i=16
    while i<len(raw)-4:
        n=int.from_bytes(raw[i:i+4],'big');number=raw[i+4]
        assert n>=5 and number not in parts
        parts[number]=raw[i:i+n];i+=n
    assert i==len(raw)-4
    return parts


def signed(b):
    """Decode the GRIB sign-magnitude scale factor."""
    n=int.from_bytes(b,'big');bits=len(b)*8
    return -(n&((1<<(bits-1))-1)) if n&(1<<(bits-1)) else n


def decode_mrms(path,crop_window):
    """Read a local MRMS 1-hour Pass 2 GRIB.gz and return its NYC crop and start hour."""
    sec=sections(gzip.decompress(Path(path).read_bytes()))
    s1,s3,s4,s5,s6,s7=[sec[k] for k in [1,3,4,5,6,7]]
    assert s1[11]==3 and int.from_bytes(s4[7:9],'big')==0
    assert s4[9:11]==bytes([6,37])
    assert int.from_bytes(s5[9:11],'big')==41 and s6[5]==255
    assert int.from_bytes(s5[5:9],'big')==7000*3500
    assert s5[19] in [8,16] and s7[5:13]==b'\x89PNG\r\n\x1a\n'
    end=datetime(int.from_bytes(s1[12:14],'big'),*s1[14:19],tzinfo=timezone.utc)
    ref=struct.unpack('>f',s5[11:15])[0];e=signed(s5[15:17]);d=signed(s5[17:19])
    col,row,width,height=map(int,crop_window)
    with Image.open(io.BytesIO(s7[5:])) as image:
        assert image.size==(7000,3500)
        pixels=np.array(image.crop((col,row,col+width,row+height)),dtype=np.float64)
    crop=((ref+pixels*2.0**e)*10.0**(-d)).astype(np.float32)
    crop[(crop<0)|~np.isfinite(crop)]=np.nan
    return crop,pd.Timestamp(end)-pd.Timedelta(hours=1),s3


def zone_rainfall(grib_files,weights_path,geometry_path):
    """Apply polygon-cell area weights, requiring at least 0.999999 valid area."""
    with np.load(weights_path) as a:weights=a['weights'];zones=a['zone_ids']
    geometry=json.loads(Path(geometry_path).read_text(encoding='utf-8'))
    rows=[];grid_definition=None
    for file in grib_files:
        crop,hour,definition=decode_mrms(file,geometry['crop_window'])
        if grid_definition is None:grid_definition=definition
        if grid_definition!=definition:raise ValueError('MRMS grid definition changed')
        values=crop.ravel();coverage=weights@np.isfinite(values).astype(float)
        means=weights@np.nan_to_num(values,nan=0)
        means[coverage<.999999]=np.nan
        for j,z in enumerate(zones):
            rows.append(dict(hour_utc=hour.isoformat(),zone_id=int(z),rain_mean_mm=means[j],valid_area_fraction=coverage[j]))
    return pd.DataFrame(rows)


def local_episodes(values,min_dry_hours=3):
    """Segment wet hours using complete dry separation and explicit missing boundaries."""
    a=np.asarray(values);wet=np.flatnonzero(np.isfinite(a)&(a>.1+EPS))
    if not len(wet):return []
    missing=np.r_[0,np.cumsum(~np.isfinite(a))]
    unknown=missing[wet[1:]]-missing[wet[:-1]+1]>0
    cut=((np.diff(wet)>=min_dry_hours+1)&~unknown)|unknown
    first=np.r_[0,np.flatnonzero(cut)+1];last=np.r_[first[1:],len(wet)]
    return [(int(wet[s]),int(wet[e-1]+1)) for s,e in zip(first,last)]


def rainfall_events(hourly,start,end):
    """Return local rainfall segments on a complete UTC grid; retain weak episodes."""
    grid=pd.date_range(start,end,tz='UTC',freq='h',inclusive='left')
    matrix=hourly.assign(hour_utc=pd.to_datetime(hourly.hour_utc,utc=True)).pivot(index='hour_utc',columns='zone_id',values='rain_mean_mm').reindex(grid)
    rows=[]
    for z in matrix:
        a=matrix[z].to_numpy()
        for left,right in local_episodes(a):
            v=a[left:right]
            rows.append(dict(zone_id=int(z),start_utc=grid[left].isoformat(),end_utc=(grid[right-1]+pd.Timedelta(hours=1)).isoformat(),
                duration_h=right-left,total_mm=float(v.sum()),peak_mm=float(v.max()),strong=bool(v.max()>=9.4-EPS),
                mean_rain_complete=bool(np.isfinite(v).all())))
    return pd.DataFrame(rows)


def features(grid):
    """Construct the fixed 173-column hour-of-week, trend and seasonal design."""
    local=grid.tz_convert(TZ);civil=local.tz_localize(None);slot=np.asarray(local.dayofweek*24+local.hour)
    angle=2*np.pi*np.asarray(local.dayofyear-1+local.hour/24)/365.25
    trend=(civil-pd.Timestamp('2023-01-01')).total_seconds().to_numpy()/86400/365.25
    return np.column_stack([np.eye(168)[slot],trend,np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])


def fit_baseline(hour_utc,actual,objective_normal,excluded,requested_indices):
    """Fit the original Ridge baseline using supplied fixed normal/exclusion masks."""
    from sklearn.linear_model import Ridge
    grid=pd.DatetimeIndex(pd.to_datetime(hour_utc,utc=True));X=features(grid)
    actual=np.asarray(actual,float);mask=np.asarray(objective_normal,bool)&~np.asarray(excluded,bool)&np.isfinite(actual)&(actual>=0)
    if mask.sum()<1000:raise ValueError('At least 1,000 eligible training hours are required')
    slots=np.asarray(grid.tz_convert(TZ).weekday*24+grid.tz_convert(TZ).hour)
    support=np.bincount(slots[mask],minlength=168)
    ix=np.asarray(requested_indices,int)
    if np.any(support[slots[ix]]==0):
        raise ValueError('Training has no support for a requested local weekday/hour cell')
    model=Ridge(alpha=1,fit_intercept=True,solver='cholesky').fit(X[mask],np.log1p(actual[mask]))
    prediction=np.maximum(np.expm1(model.predict(X[ix])),0)
    return prediction


def centered_reference(raw_gaps,dates,clock_cells,requested_clock_cells):
    """Return equal-date mean normal residual ratios for matching local clock classes."""
    q=pd.DataFrame({'date':dates,'clock':clock_cells,'gap':raw_gaps})
    means=q.groupby(['clock','date']).gap.mean().groupby('clock').mean()
    return means.reindex(requested_clock_cells).to_numpy()


def prepare_raw(config_path,input_root,output_dir):
    """Execute the explicitly listed local preprocessing jobs; no downloads are performed."""
    root=Path(input_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'));zone_ids=pd.read_csv(root/cfg['zone_labels']).zone_id
    audits=[]
    for job in cfg.get('trip_jobs',[]):
        frame,audit=aggregate_trips(root/job['file'],job['source'],job['month'],zone_ids)
        name=f"{job['source']}_{job['month']}_hourly.csv.gz"
        frame.to_csv(out/name,index=False,compression={'method':'gzip','mtime':0})
        audits.append(dict(file=job['file'],output=name,**audit))
    rain=cfg.get('rainfall')
    if rain:
        paths=[root/name for name in rain['files']]
        frame=zone_rainfall(paths,root/rain['weights'],root/rain['geometry'])
        frame.to_csv(out/'zone_hourly_rainfall.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        if 'study_start' in rain and 'study_end_exclusive' in rain:
            rainfall_events(frame,rain['study_start'],rain['study_end_exclusive']).to_csv(out/'local_rainfall_events.csv',index=False)
    for job in cfg.get('baseline_jobs',[]):
        with np.load(root/job['file']) as a:
            prediction=fit_baseline(a['hour_utc'],a['actual'],a['objective_normal'],a['excluded'],a['requested_indices'])
            pd.DataFrame({'hour_utc':a['hour_utc'][a['requested_indices']],'prediction':prediction}).to_csv(out/job['output'],index=False)
    (out/'preprocessing_summary.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
