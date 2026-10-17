"""Reader for TA Instruments Affinity ITC / Nano ITC ``.nitc`` files (ITCRun).

A ``.nitc`` file is a gzip-compressed .NET ``BinaryFormatter`` stream of a
``DataSet`` with (at least) these tables:

* ``DataPointsTable``    Time [s], HeatRate [uJ/s], temperatures, DAC readings
* ``InjectionsTable``    StartTime, EndTime [s], Size [uL], InjectionInterval
* ``BaselineNodesTable`` X [s], Y [uJ/s] (offset-subtracted baseline nodes)
* ``KVP``                run parameters as key/value strings
                         (concentrations in mM, volumes in uL, temperature in C)
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import nrbf


@dataclass
class ITCRun:
    path: Path
    time: np.ndarray                 # s
    heat_rate: np.ndarray            # uJ/s, heatRateOffset subtracted
    inj_start: np.ndarray            # s
    inj_end: np.ndarray              # s
    inj_volume: np.ndarray           # uL
    cell_conc: float                 # mM (titrate, "M" in the cell)
    syringe_conc: float              # mM (titrant, "X" in the syringe)
    cell_volume: float               # uL
    temperature: float               # C
    exotherm_up: bool = True         # True: exothermic peaks are positive in heat_rate
    baseline_nodes: tuple[np.ndarray, np.ndarray] | None = None
    params: dict[str, str] = field(default_factory=dict)
    tables: dict[str, dict[str, np.ndarray | list]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def n_injections(self) -> int:
        return len(self.inj_start)


def read_tables(path: str | Path) -> dict[str, dict[str, np.ndarray | list]]:
    """Return every DataTable in a ``.nitc`` file as ``{table: {column: values}}``."""
    raw = Path(path).read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # not compressed; try the stream as-is
    root = nrbf.loads(raw)
    if root is None or "DataSet.Tables.Count" not in root.members:
        raise ValueError(f"{path}: not a serialized ITC DataSet")

    tables: dict[str, dict[str, np.ndarray | list]] = {}
    for t in range(root["DataSet.Tables.Count"]):
        schema = nrbf.loads(root[f"DataSet.Tables_{t}"])
        name = schema["DataTable.TableName"]
        n_rows = root[f"DataTable_{t}.Rows.Count"]
        records = root[f"DataTable_{t}.Records"]["_items"]
        cols: dict[str, np.ndarray | list] = {}
        for c in range(schema["DataTable.Columns.Count"]):
            col_name = schema[f"DataTable.DataColumn_{c}.ColumnName"]
            values = records[c] if c < len(records) and records[c] is not None else []
            cols[col_name] = values[:n_rows]
        tables[name] = cols
    return tables


def _num(params: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(params[key])
    except (KeyError, ValueError):
        return default


def read_nitc(path: str | Path) -> ITCRun:
    path = Path(path)
    tables = read_tables(path)
    kvp = tables.get("KVP", {})
    params = dict(zip(kvp.get("Key", []), kvp.get("Value", [])))

    data = tables["DataPointsTable"]
    time = np.asarray(data["Time"], dtype=float)
    heat = np.asarray(data["HeatRate"], dtype=float) - _num(params, "heatRateOffset", 0.0)

    inj = tables["InjectionsTable"]
    baseline = tables.get("BaselineNodesTable")
    nodes = None
    if baseline is not None and len(baseline.get("X", [])):
        nodes = (np.asarray(baseline["X"], float), np.asarray(baseline["Y"], float))

    return ITCRun(
        path=path,
        time=time,
        heat_rate=heat,
        inj_start=np.asarray(inj["StartTime"], float),
        inj_end=np.asarray(inj["EndTime"], float),
        inj_volume=np.asarray(inj["Size"], float),
        cell_conc=_num(params, "cellconcentration"),
        syringe_conc=_num(params, "syringeconcentration"),
        cell_volume=_num(params, "cellvolume"),
        temperature=_num(params, "tempsetpoint", 25.0),
        exotherm_up=params.get("ExothermUp", "True").strip().lower() == "true",
        baseline_nodes=nodes,
        params=params,
        tables=tables,
    )
