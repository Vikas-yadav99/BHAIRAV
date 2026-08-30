#!/usr/bin/env python3
"""BHAIRAV Crowd Surveillance Runner"""
import json, sys, time, signal
from pathlib import Path
from collections import Counter

CROWD_DIR = Path('src/bhairav/test_footage/heavy_crowd')
OUTPUT_DIR = CROWD_DIR / 'reports'
OUTPUT_DIR.mkdir(exist_ok=True)
ALERTS_JSONL = OUTPUT_DIR / 'alerts_live.jsonl'
TRAJ_JSONL = OUTPUT_DIR / 'trajectories.jsonl'
RESULTS_JSONL = OUTPUT_DIR / 'clip_results.jsonl'
REPORT_JSON = OUTPUT_DIR / 'surveillance_report.json'

def load_crowd_clips():
    p = CROWD_DIR / 'crowd_clips.json'
    if p.exists():
        return json.loads(p.read_text())
    return [{'path': str(x), 'file': x.name} for x in sorted(CROWD_DIR.glob('clip_*.mp4'))]

def ajl(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')

NL = chr(10)
EQ = '=' * 70

def save_report(tf, tp, ta, rc, sc, st, cr):
    el = time.time() - st
    ad = []
    if ALERTS_JSONL.exists():
        for l in ALERTS_JSONL.read_text().splitlines():
            if l.strip():
                ad.append(json.loads(l))
    rpt = {
        'run_info': {'end': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': round(el, 1), 'clips': len(cr)},
        'totals': {'frames': tf, 'persons': tp, 'alerts': ta, 'fps': round(tf / el, 1) if el > 0 else 0,
                   'video_s': round(sum(c.get('duration_sec', 0) for c in cr), 1)},
        'alerts_detail': {'total': ta, 'by_rule': dict(rc.most_common()), 'by_sev': dict(sc.most_common()), 'list': ad[:200]},
        'clip_results': cr,
    }
    REPORT_JSON.write_text(json.dumps(rpt, indent=2, default=str))
    print(NL + EQ)
    print('COMPLETE: %d clips, %d frames, %d persons, %d alerts' % (len(cr), tf, tp, ta))
    if el > 0:
        print('Speed: %.1f fps, Time: %.1f min' % (tf / el, el / 60))
    print('By rule: %s' % dict(rc.most_common()))
    print('By severity: %s' % dict(sc.most_common()))
    print('Report: %s' % REPORT_JSON)


def run():
    import cv2
    from bhairav.config import AppConfig
    from bhairav.pipeline import build_engine, make_detector, run_pipeline
    from bhairav.face_tracking import TrajectoryPredictor
    from bhairav.alert_log import AlertLog

    cc = load_crowd_clips()
    lc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ac = cc * lc
    print('BHAIRAV CROWD SURVEILLANCE: %d clips x %d loops = %d runs' % (len(cc), lc, len(ac)))
    for f in [ALERTS_JSONL, TRAJ_JSONL, RESULTS_JSONL]:
        f.unlink(missing_ok=True)

    cfg = AppConfig()
    cfg.detector = 'yolo'
    cfg.model.conf = 0.25
    cfg.model.imgsz = 480

    al = AlertLog(ALERTS_JSONL)
    trk = TrajectoryPredictor(zones=cfg.zones, persist_path=TRAJ_JSONL)
    cr = []
    rc = Counter()
    sc = Counter()
    tf = 0
    tp_total = 0
    ta = 0
    st = time.time()

    def sig_handler(s, f):
        print(NL + 'Interrupted after %d clips' % len(cr))
        save_report(tf, tp_total, ta, rc, sc, st, cr)
        trk.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    for i, ci in enumerate(ac):
        cp = Path(ci['path'])
        if not cp.exists():
            continue
        cm = 'CAM-%03d' % (i + 1)
        cn = ci['file']
        print(NL + '--- %d/%d: %s (%s) ---' % (i + 1, len(ac), cn, cm))

        cap = cv2.VideoCapture(str(cp))
        if not cap.isOpened():
            print('  SKIP')
            continue
        cframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        dur = cframes / cfps if cfps > 0 else 0

        cs = time.time()
        frame_count = [0]
        pids = set()
        eng = build_engine(cfg)

        def on_frame(state, alerts, _fc=frame_count, _pids=pids):
            _fc[0] += 1
            for t in state.tracks:
                if t.label == 'person':
                    _pids.add(t.track_id)
                    if t.bbox:
                        x1, y1, x2, y2 = t.bbox
                        cx = (x1 + x2) / 2.0 / state.frame_w if state.frame_w else 0.5
                        cy = (y1 + y2) / 2.0 / state.frame_h if state.frame_h else 0.5
                        trk.update(person_id=cm + '-P' + str(t.track_id),
                                   x=cx, y=cy, camera_id=cm,
                                   frame_id=state.frame_id, timestamp=state.timestamp)
            if _fc[0] % 50 == 0:
                print('  F%d/%d: %d persons' % (_fc[0], cframes, len(_pids)))
            return None

        try:
            det = make_detector(cfg, detector='yolo', source=str(cp))
            alerts = run_pipeline(det, eng, source=str(cp), on_frame=on_frame)
            ce = time.time() - cs
            fa = frame_count[0] / ce if ce > 0 else 0
            for a in alerts:
                al.write(a)
                rc[a.rule] += 1
                sc[a.severity.value] += 1
            tf += frame_count[0]
            tp_total += len(pids)
            ta += len(alerts)
            r = {'clip': cn, 'camera': cm, 'duration_sec': round(dur, 1),
                 'frames': frame_count[0], 'fps': round(fa, 1),
                 'persons': len(pids), 'alerts': len(alerts), 'time': round(ce, 1),
                 'rules': dict(Counter(a.rule for a in alerts))}
            ajl(RESULTS_JSONL, r)
            cr.append(r)
            print('  DONE: %d persons, %d alerts, %.1f fps' % (len(pids), len(alerts), fa))
            for a in alerts[:3]:
                print('    [%s] %s: %s' % (a.severity.value, a.rule, a.message))
            if len(alerts) > 3:
                print('    ...+%d more' % (len(alerts) - 3))
        except Exception as e:
            print('  FAILED: %s' % e)
            ajl(RESULTS_JSONL, {'clip': cn, 'camera': cm, 'error': str(e)})

    save_report(tf, tp_total, ta, rc, sc, st, cr)
    trk.shutdown()

if __name__ == '__main__':
    run()
