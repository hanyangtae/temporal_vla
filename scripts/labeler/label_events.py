"""Backward-compatible point/interval validation and flat event exports."""
import copy, math

EVENT_COLS=['rel','key','s','j','n','succ','sig','no_failure','mark_index','kind','type','t','frame','start_t','start_frame','end_t','end_frame','fps','frames_total','mark_note','note','by','updated_at']

def validate_record(rec):
    rec=copy.deepcopy(rec)
    marks=rec.get('marks') or []
    if not isinstance(marks,list):raise ValueError('marks must be a list')
    fps=float(rec.get('fps') or 20)
    if not math.isfinite(fps) or fps<=0:raise ValueError('invalid fps')
    if rec.get('no_failure') and marks:raise ValueError('no_failure cannot erase event marks implicitly')
    limit=rec.get('frames_total')
    for m in marks:
        if not isinstance(m,dict):raise ValueError('invalid mark')
        kind=m.get('kind','point')
        if kind not in ('point','interval'):raise ValueError('unknown mark kind')
        def boundary(t,f):
            if isinstance(m.get(t),bool) or isinstance(m.get(f),bool):raise ValueError('invalid boundary')
            tv=float(m[t]);fv=float(m[f])
            if not math.isfinite(tv) or not math.isfinite(fv) or tv<0 or fv<0 or not fv.is_integer():raise ValueError('invalid boundary')
            if abs(tv-fv/fps)>0.001:raise ValueError('time/frame mismatch')
            if limit is not None and fv>float(limit):raise ValueError('boundary beyond video')
        boundary('t','frame')
        if kind=='interval':
            boundary('start_t','start_frame');boundary('end_t','end_frame')
            if m['t']!=m['start_t'] or m['frame']!=m['start_frame']:raise ValueError('start alias mismatch')
            if m['end_frame']<=m['start_frame']:raise ValueError('end must follow start')
    first=min((m for m in marks if m.get('type')!='successful_retry'),key=lambda m:m['t'],default=None)
    rec.update(t_fail=first['t'] if first else None,frame=first['frame'] if first else None,type=first.get('type') if first else None,n_marks=len(marks))
    return rec

def event_rows(records):
    for r in sorted(records,key=lambda r:r['rel']):
        marks=r.get('marks') or []
        for i,m in enumerate(marks or [None],1):
            out={k:r.get(k,'') for k in EVENT_COLS}
            out.update(mark_index=i if m else '',kind=(m.get('kind','point') if m else 'none'),type=m.get('type') if m else '',t=m.get('t') if m else '',frame=m.get('frame') if m else '',start_t=m.get('start_t',m.get('t')) if m else '',start_frame=m.get('start_frame',m.get('frame')) if m else '',end_t=m.get('end_t','') if m else '',end_frame=m.get('end_frame','') if m else '',mark_note=m.get('note','') if m else '')
            yield out
