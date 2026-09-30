"""EUMETSAT Data Store access (eumdac): SEVIRI Level 1.5, FCI Level 1c and CLM for one time slot.

Credentials: the consumer key and secret from https://api.eumetsat.int/api-key/ in the environment
variables EUMETSAT_KEY and EUMETSAT_SECRET. They are never printed or written."""
import os
import shutil
import time
from datetime import datetime, timedelta

SLOT_PROBE_MIN = 5.0          # a product matches a slot if its sensing interval covers slot + 5 min
_DS = {}


def slot_dt(slot):
    return datetime.strptime(slot, "%Y%m%d%H")


def datastore():
    if "ds" not in _DS:
        import eumdac
        key, sec = os.environ.get("EUMETSAT_KEY"), os.environ.get("EUMETSAT_SECRET")
        if not key or not sec:
            raise SystemExit("set EUMETSAT_KEY and EUMETSAT_SECRET (consumer key and secret from "
                             "https://api.eumetsat.int/api-key/)")
        tok = eumdac.AccessToken((key, sec))
        _ = tok.access_token
        _DS["ds"] = eumdac.DataStore(tok)
    return _DS["ds"]


def search(collection, slot, before_min=20, after_min=35):
    t = slot_dt(slot)
    coll = datastore().get_collection(collection)
    return list(coll.search(dtstart=t - timedelta(minutes=before_min), dtend=t + timedelta(minutes=after_min)))


def pick_product(products, slot, tag, needle=None):
    """The product of the requested slot: platform tag (and needle) in its id and a sensing interval that
    covers slot + SLOT_PROBE_MIN; otherwise the nearest start within -2..+10 min."""
    t0 = slot_dt(slot)
    probe = t0 + timedelta(minutes=SLOT_PROBE_MIN)
    cands = [p for p in products if tag in str(p) and (needle is None or needle in str(p))]
    cover = [p for p in cands if p.sensing_start <= probe <= p.sensing_end]
    if cover:
        return min(cover, key=lambda p: abs((p.sensing_start - t0).total_seconds()))
    near = [p for p in cands if -120 <= (p.sensing_start - t0).total_seconds() <= 600]
    return min(near, key=lambda p: abs((p.sensing_start - t0).total_seconds())) if near else None


def entries_seviri(entries):
    return [e for e in entries if e.lower().endswith(".nat")]


def entries_fci(entries):
    return [e for e in entries if "-BODY-" in os.path.basename(e) and e.endswith(".nc")]


def entries_clm(entries):
    return [e for e in entries if e.lower().endswith((".grb", ".grib", ".grb2"))
            or ("MSGCLMK" in os.path.basename(e) and not e.lower().endswith((".xml", ".jpg", ".png")))]


def resolve(slot, cfg):
    """{'seviri'|'fci'|'clm': (product, entries)} for one YYYYMMDDHH slot; raises LookupError if missing."""
    ds = cfg["datastore"]
    got = {}
    for key, coll, tag, needle, pick in (
            ("seviri", ds["collection_seviri"], ds["seviri_tag"], None, entries_seviri),
            ("fci", ds["collection_fci"], ds["fci_tag"], "-FD-", entries_fci),
            ("clm", ds["collection_clm"], ds["seviri_tag"], None, entries_clm)):
        p = pick_product(search(coll, slot), slot, tag, needle)
        if p is None:
            raise LookupError(f"{slot}: no {key} product in {coll} for {tag}")
        ents = pick(list(p.entries))
        if not ents:
            raise LookupError(f"{slot}: {key} product {p} has no usable entries")
        got[key] = (p, ents)
    return got


def fetch(product, entries, dest_dir, retries=4):
    """Download entries into dest_dir; skips complete files, never leaves partial ones."""
    os.makedirs(dest_dir, exist_ok=True)
    out = []
    for e in entries:
        fn = os.path.join(dest_dir, os.path.basename(e))
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            out.append(fn)
            continue
        for attempt in range(1, retries + 1):
            try:
                with product.open(entry=e) as src, open(fn + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 << 20)
                os.replace(fn + ".part", fn)
                break
            except Exception as exc:                     # network hiccups: retry with back-off
                print(f"  {os.path.basename(e)[:60]} attempt {attempt}: {type(exc).__name__}: {exc}")
                if os.path.exists(fn + ".part"):
                    os.remove(fn + ".part")
                time.sleep(15 * attempt)
        else:
            raise RuntimeError(f"could not download {e}")
        out.append(fn)
    return out
