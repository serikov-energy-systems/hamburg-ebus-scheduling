"""Build bus trip instances from an HVV GTFS feed (published by Hamburg Transparenzportal).

Only used to create the small instance files in data/instances; the solvers read those files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R_EARTH = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def parse_time(s: pd.Series) -> pd.Series:
    """GTFS times 'H:MM:SS' (hours may exceed 23) to minutes after midnight."""
    p = s.str.split(":", expand=True).astype(int)
    return p[0] * 60 + p[1] + p[2] / 60.0


def active_services(gtfs_dir: str, date: str) -> set[str]:
    cal = pd.read_csv(f"{gtfs_dir}/calendar.txt", dtype=str)
    cd = pd.read_csv(f"{gtfs_dir}/calendar_dates.txt", dtype=str)
    wd = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][
        pd.Timestamp(date).weekday()
    ]
    ds = date.replace("-", "")
    act = set(cal[(cal.start_date <= ds) & (cal.end_date >= ds) & (cal[wd] == "1")].service_id)
    add = set(cd[(cd.date == ds) & (cd.exception_type == "1")].service_id)
    rem = set(cd[(cd.date == ds) & (cd.exception_type == "2")].service_id)
    return (act | add) - rem


def build_trips(gtfs_dir: str, date: str, lines: list[str]) -> pd.DataFrame:
    """One row per bus trip of the given line numbers on the given day."""
    routes = pd.read_csv(f"{gtfs_dir}/routes.txt", dtype=str)
    routes = routes[(routes.route_type == "3") & routes.route_short_name.isin(lines)]
    trips = pd.read_csv(f"{gtfs_dir}/trips.txt", dtype=str)
    trips = trips[trips.route_id.isin(routes.route_id) & trips.service_id.isin(active_services(gtfs_dir, date))]
    trips = trips.merge(routes[["route_id", "route_short_name"]], on="route_id")
    st = pd.read_csv(
        f"{gtfs_dir}/stop_times.txt",
        dtype={"trip_id": str, "stop_id": str},
        usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    )
    st = st[st.trip_id.isin(trips.trip_id)].sort_values(["trip_id", "stop_sequence"])
    stops = pd.read_csv(f"{gtfs_dir}/stops.txt", dtype={"stop_id": str})
    stops["lat"] = stops.stop_lat.astype(float)
    stops["lon"] = stops.stop_lon.astype(float)
    # terminal = all platforms with the same name share one charger site and one location
    name_loc = stops.groupby("stop_name")[["lat", "lon"]].mean()
    stops = stops.join(name_loc, on="stop_name", rsuffix="_n")
    st = st.merge(stops[["stop_id", "stop_name", "lat_n", "lon_n"]], on="stop_id")
    st = st.sort_values(["trip_id", "stop_sequence"])
    first = st.groupby("trip_id").first()
    last = st.groupby("trip_id").last()
    # length along the shapes; falls back to the stop chain
    shapes = pd.read_csv(f"{gtfs_dir}/shapes.txt", usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], dtype={"shape_id": str})
    shapes = shapes[shapes.shape_id.isin(trips.shape_id)].sort_values(["shape_id", "shape_pt_sequence"])
    d = haversine_km(
        shapes.shape_pt_lat.values[:-1], shapes.shape_pt_lon.values[:-1],
        shapes.shape_pt_lat.values[1:], shapes.shape_pt_lon.values[1:],
    )
    same = (shapes.shape_id.values[:-1] == shapes.shape_id.values[1:])
    seglen = pd.Series(np.where(same, d, 0.0), index=shapes.shape_id.values[:-1])
    shape_km = seglen.groupby(level=0).sum()
    out = trips.set_index("trip_id")[["route_short_name", "direction_id", "shape_id"]].copy()
    out["start_min"] = parse_time(first.departure_time)
    out["end_min"] = parse_time(last.arrival_time)
    out["start_name"] = first.stop_name
    out["end_name"] = last.stop_name
    out["start_lat"], out["start_lon"] = first.lat_n, first.lon_n
    out["end_lat"], out["end_lon"] = last.lat_n, last.lon_n
    out["length_km"] = out.shape_id.map(shape_km)
    n_stops = st.groupby("trip_id").size()
    out["n_stops"] = n_stops
    out = out.dropna(subset=["start_min", "end_min", "length_km"]).reset_index()
    out = out[(out.end_min > out.start_min) & (out.length_km > 0.5)]
    out = out.sort_values(["start_min", "end_min"]).reset_index(drop=True)
    out["trip_id"] = out.trip_id.astype(str)
    return out.drop(columns=["shape_id"])
