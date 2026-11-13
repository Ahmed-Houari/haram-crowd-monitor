"""Alert engine (step 7): crowd and road alerts with a cooldown, so a flickering condition does not spam.

Rules (crowd rules since 2026-10-27 on the Mataf's people per m^2 and the crowd-safety levels of the dashboard,
Fruin's walkway level of service, backend/aiview.py LEVELS; before: a camera's top 20 % of its own history):
  crowd_now       the Mataf's reading (ring 1-25 m, mean of its last MATAF_AVG_MIN minutes, the newest at most
                  FRESH_MIN old) is crowded (>= CROWDED_M2: watch) or packed (>= PACKED_M2: high).
  crowd_forecast  the forecast (edge/forecast.py) for the Mataf reaches crowded / packed within 30 min.
                  Needs a trained model (models/forecast) that covers the Mataf.
  road            congestion within 1 km >= ROAD_PCT % of road length on each of the last ROAD_POLLS TomTom polls
                  (one high poll is not enough); "high" from ROAD_HIGH_PCT %.
Life of an alert: it opens when its condition becomes true and stays open while it holds; it closes once the
condition has been false for HOLD_MIN. If the condition returns within COOLDOWN_MIN after the close, the same
alert re-opens instead of a new one. State lives in memory (the API process); after a restart, open conditions
simply open fresh alerts.

Self-test:  .venv/Scripts/python -m edge.alerts --selftest
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

FRESH_MIN = 20
HOLD_MIN = 10
COOLDOWN_MIN = 30
KEEP_HOURS = 24  # closed alerts stay listed this long
ROAD_PCT, ROAD_HIGH_PCT, ROAD_POLLS = 25.0, 40.0, 2
FORECAST_WITHIN_MIN = 30
CROWDED_M2, PACKED_M2 = 1.1, 2.2  # people per m^2: crowded (Fruin LOS E) / packed (LOS F)
MATAF_AVG_MIN = 5


@dataclass
class Condition:
    """One rule evaluated for one place, now."""
    kind: str  # crowd_now | crowd_forecast | road
    key: str  # camera id or road area
    on: bool
    severity: str = "watch"  # watch | high
    place: str = ""
    message: str = ""
    value: float | None = None
    threshold: float | None = None


@dataclass
class Alert:
    id: str
    kind: str
    key: str
    severity: str
    place: str
    message: str
    start_utc: str
    last_on_utc: str
    end_utc: str | None = None
    value: float | None = None
    peak: float | None = None
    threshold: float | None = None
    reopened: int = 0

    @property
    def active(self) -> bool:
        return self.end_utc is None


@dataclass
class AlertEngine:
    hold_min: float = HOLD_MIN
    cooldown_min: float = COOLDOWN_MIN
    alerts: list[Alert] = field(default_factory=list)
    _seq: int = 0

    def _latest(self, kind: str, key: str) -> Alert | None:
        for a in reversed(self.alerts):
            if a.kind == kind and a.key == key:
                return a
        return None

    def update(self, now: datetime, conditions: list[Condition]) -> list[Alert]:
        """Apply this round's conditions; returns the open + recently closed alerts (see view() for the order)."""
        iso = now.isoformat(timespec="seconds")
        seen = set()
        for c in conditions:
            seen.add((c.kind, c.key))
            a = self._latest(c.kind, c.key)
            if c.on:
                if a and a.active:
                    a.last_on_utc, a.value, a.message, a.severity = iso, c.value, c.message, _max_sev(a.severity, c.severity)
                    a.peak = _peak(a.peak, c.value)
                elif a and now - _dt(a.end_utc) < timedelta(minutes=self.cooldown_min):
                    a.end_utc, a.last_on_utc, a.value, a.message = None, iso, c.value, c.message  # back within cooldown
                    a.peak, a.reopened = _peak(a.peak, c.value), a.reopened + 1
                else:
                    self._seq += 1
                    self.alerts.append(Alert(f"{c.kind}:{c.key}:{self._seq}", c.kind, c.key, c.severity, c.place,
                                             c.message, iso, iso, None, c.value, c.value, c.threshold))
            elif a and a.active and now - _dt(a.last_on_utc) >= timedelta(minutes=self.hold_min):
                a.end_utc = iso
        for a in self.alerts:  # a rule that is no longer evaluated (camera gone) closes after the hold time too
            if a.active and (a.kind, a.key) not in seen and now - _dt(a.last_on_utc) >= timedelta(minutes=self.hold_min):
                a.end_utc = iso
        cutoff = now - timedelta(hours=KEEP_HOURS)
        self.alerts = [a for a in self.alerts if a.active or _dt(a.end_utc) >= cutoff]
        return self.alerts

    def view(self) -> list[dict]:
        """Open alerts first (newest first), then closed ones (newest first)."""
        open_ = sorted((a for a in self.alerts if a.active), key=lambda a: a.start_utc, reverse=True)
        closed = sorted((a for a in self.alerts if not a.active), key=lambda a: a.end_utc or "", reverse=True)
        return [asdict(a) | {"active": a.active} for a in open_ + closed]


def _dt(iso: str | None) -> datetime:
    return datetime.fromisoformat(iso) if iso else datetime.min.replace(tzinfo=timezone.utc)


def _peak(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


def _max_sev(a: str, b: str) -> str:
    return "high" if "high" in (a, b) else "watch"


def _minutes_since(iso: str, now: datetime) -> float:
    return (now - datetime.fromisoformat(iso)).total_seconds() / 60


def _level(v: float) -> tuple[str, str, float] | None:
    """(level, severity, threshold) when v is crowded or packed, else None."""
    if v >= PACKED_M2:
        return "packed", "high", PACKED_M2
    if v >= CROWDED_M2:
        return "crowded", "watch", CROWDED_M2
    return None


def mataf_conditions(mataf: dict | None, forecast: dict | None, now: datetime) -> list[Condition]:
    """crowd_now + crowd_forecast conditions for the Mataf (people per m^2, ring 1-25 m) from /api/mataf/now and
    edge.forecast.predict_now()."""
    out = []
    if mataf and mataf.get("minute_utc") and _minutes_since(mataf["minute_utc"], now) <= FRESH_MIN:
        last = _dt(mataf["minute_utc"])
        vals = [v for t, v in mataf.get("spark") or [] if (last - _dt(t)).total_seconds() < MATAF_AVG_MIN * 60]
        v = sum(vals) / len(vals) if vals else mataf["people_m2"]
        lv = _level(v)
        out.append(Condition("crowd_now", "mataf", lv is not None, lv[1] if lv else "watch", "Mataf",
                             f"Mataf is {lv[0]}: {v:.2f} people per m² ({MATAF_AVG_MIN}-min mean, 1-25 m from the "
                             f"Kaaba; {lv[0]} from {lv[2]:.1f})" if lv else "", round(v, 3), lv[2] if lv else CROWDED_M2))
    for f in (forecast or {}).get("cameras", []) if (forecast or {}).get("available") else []:
        if (f.get("view_id"), f.get("zone")) != ("mataf", "all"):
            continue
        soon = [(p, _level(p["density"])) for p in f["predictions"] if p["h"] <= FORECAST_WITHIN_MIN]
        hits = [(p, lv) for p, lv in soon if lv is not None]
        p, lv = max(hits, key=lambda x: x[0]["density"]) if hits else (None, None)
        when = datetime.fromisoformat(p["utc"]).astimezone(timezone(timedelta(hours=3))).strftime("%H:%M") if p else ""
        out.append(Condition("crowd_forecast", "mataf", p is not None, lv[1] if lv else "watch", "Mataf",
                             f"Mataf expected {lv[0]} by {when}: forecast {p['density']:.2f} people per m² "
                             f"({lv[0]} from {lv[2]:.1f})" if p else "", p["density"] if p else None,
                             lv[2] if lv else CROWDED_M2))
    return out


def road_conditions(series: list[dict], now: datetime, area: str = "r1km") -> list[Condition]:
    """road condition from /api/traffic/series rows (all roads): the last ROAD_POLLS polls must all be high."""
    polls = sorted((p for p in series if p["area"] == area and p.get("congested_pct") is not None),
                   key=lambda p: p["poll_utc"])[-ROAD_POLLS:]
    if len(polls) < ROAD_POLLS or _minutes_since(polls[-1]["poll_utc"], now) > 15:
        return []
    low = min(p["congested_pct"] for p in polls)
    on = low >= ROAD_PCT
    span = round(_minutes_since(polls[0]["poll_utc"], now))
    return [Condition("road", area, on, "high" if low >= ROAD_HIGH_PCT else "watch", "Roads within 1 km",
                      f"Roads within 1 km: {round(polls[-1]['congested_pct'])} % of road length congested "
                      f"(over {ROAD_PCT:.0f} % for {span} min)", polls[-1]["congested_pct"], ROAD_PCT)]


def selftest() -> None:
    """A flickering road condition and a crowd episode: check open / hold / cooldown / re-open / new alert."""
    eng = AlertEngine()
    t0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    pattern = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    for i, on in enumerate(pattern):  # one evaluation every 5 min
        eng.update(t0 + timedelta(minutes=5 * i), [Condition("road", "r1km", bool(on), "watch", "Roads", "msg", 30.0, 25.0)])
    roads = [a for a in eng.alerts if a.kind == "road"]
    # minute 0-15 flicker -> one alert; closes 10 min after the last "on" (min 15 -> closed at 25);
    # back at min 70 (45 min after the close > cooldown 30) -> a NEW alert; ... back at min 170 -> another new one
    assert len(roads) == 3, [asdict(a) for a in roads]
    assert roads[0].end_utc == (t0 + timedelta(minutes=25)).isoformat(timespec="seconds"), roads[0].end_utc
    eng2 = AlertEngine()
    for i, on in enumerate([1, 0, 0, 0, 1]):  # off for 15 min (> hold 10) then back 10 min after close (< cooldown)
        eng2.update(t0 + timedelta(minutes=5 * i), [Condition("crowd_now", "cam0", bool(on), "high", "z", "m", 1.0, 1.0)])
    assert len(eng2.alerts) == 1 and eng2.alerts[0].reopened == 1 and eng2.alerts[0].active, [asdict(a) for a in eng2.alerts]
    cond = road_conditions([{"area": "r1km", "poll_utc": (t0 - timedelta(minutes=5)).isoformat(), "congested_pct": 31},
                            {"area": "r1km", "poll_utc": t0.isoformat(), "congested_pct": 44}], t0)
    assert cond and cond[0].on and cond[0].severity == "watch", cond
    mataf = {"people_m2": 1.3, "minute_utc": t0.isoformat(),
             "spark": [[(t0 - timedelta(minutes=m)).isoformat(), v] for m, v in ((4, 1.0), (2, 1.2), (0, 1.3))]}
    cc = mataf_conditions(mataf, None, t0)
    assert len(cc) == 1 and cc[0].on and cc[0].severity == "watch" and "crowded" in cc[0].message, cc
    cc = mataf_conditions({**mataf, "spark": [[t0.isoformat(), 2.5]], "people_m2": 2.5}, None, t0)
    assert cc[0].on and cc[0].severity == "high" and "packed" in cc[0].message, cc
    assert not mataf_conditions({**mataf, "spark": [[t0.isoformat(), 0.6]]}, None, t0)[0].on
    print(f"alerts selftest OK: flicker -> {len(roads)} road alerts (1 per episode), re-open within cooldown works, "
          "the Mataf's crowd-safety levels and road polls")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
