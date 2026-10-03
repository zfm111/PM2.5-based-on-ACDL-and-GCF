"""Convert ERA5 GRIB files into streaming MATLAB v7.3 MAT files.

Each output contains an ECMWF struct. Data uses MATLAB dimensions
[lat, lon, time] for single levels and [lat, lon, level, time] for pressure
levels. The writer streams along time, so monthly pressure fields are not
materialized as a >4 GiB in-memory array.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import glob
import os
import sys

import cfgrib
import h5py
import hdf5storage
import numpy as np
import xarray as xr


# ===== Configuration =====
BASE_DIR = Path(r"D:/era5_data")
OUTPUT_BASE = Path(r"D:/era5_mat")
TIME_CHUNK_HOURS = 24
GZIP_LEVEL = 4
OVERWRITE = False
DRY_RUN = False

EXPECTED_PRESSURE_LEVELS = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175,
    150, 125, 100, 70, 50, 30, 20, 10,
], dtype=np.float32)

LAND_VARIABLES = {
    # cfgrib 用 ECMWF 短名作变量名：2t→t2m、10u→u10、10v→v10、2d→d2m
    "t2m": ("2m_temperature", "K"),
    "u10": ("10m_u_component_of_wind", "m s^-1"),
    "v10": ("10m_v_component_of_wind", "m s^-1"),
    "d2m": ("2m_dewpoint_temperature", "K"),
    "sp": ("surface_pressure", "Pa"),
}
PRESSURE_VARIABLES = {
    "r": ("relative_humidity", "%"),
    "t": ("temperature", "K"),
    "z": ("geopotential", "m^2 s^-2"),
}
SINGLE_VARIABLES = {
    "z": ("geopotential", "m^2 s^-2"),
    "tcwv": ("total_column_water_vapour", "kg m^-2"),
    "blh": ("boundary_layer_height", "m"),
}


def to_datenum(times: np.ndarray) -> np.ndarray:
    """numpy datetime64 -> MATLAB datenum (UTC)."""
    dt = np.asarray(times, dtype="datetime64[ns]")
    return (dt.astype("int64").astype(np.float64) / 8.64e13 + 719529.0).reshape(-1, 1)


def coordinate_names(da: xr.DataArray) -> tuple[str, str, str]:
    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"
    time = "time" if "time" in da.dims else "valid_time" if "valid_time" in da.dims else None
    if time is None:
        raise KeyError(f"{da.name} 缺少 time/valid_time 维度：{da.dims}")
    return lat, lon, time


def vertical_dim(da: xr.DataArray) -> str | None:
    lat, lon, time = coordinate_names(da)
    extra = [d for d in da.dims if d not in (lat, lon, time)]
    if len(extra) > 1:
        raise ValueError(f"{da.name} 具有异常维度：{da.dims}")
    return extra[0] if extra else None


def open_grib(path: Path) -> xr.Dataset | None:
    try:
        try:
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        except Exception:
            groups = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
            if not groups:
                raise RuntimeError("cfgrib 未返回 Dataset")
            ds = xr.merge(groups, compat="override", join="exact")
        if not ds.data_vars:
            raise RuntimeError("GRIB 中未发现变量")
        print(f"    [OK] {path.name}: {list(ds.data_vars)}")
        return ds
    except Exception as exc:
        print(f"    [ERROR] 无法打开 {path.name}: {exc}")
        return None


def pressure_part_levels(ds: xr.Dataset, source_var: str) -> np.ndarray:
    """返回某 part 中 source_var 的气压层值(hPa, float32)。"""
    vert = vertical_dim(ds[source_var])
    return np.asarray(ds[source_var][vert].values, dtype=np.float32)


def validate_pressure_parts(datasets: list[xr.Dataset], source_var: str) -> tuple[bool, str]:
    """不拼接、逐 part 校验:变量齐全、lat/lon/time 一致、气压层并集==32 层(1000→10)。"""
    if not datasets:
        return False, "无 pressure part"
    valid = [ds for ds in datasets if ds is not None and source_var in ds]
    if len(valid) != len(datasets):
        return False, f"{source_var} 在部分 part 中缺失"
    da0 = valid[0][source_var]
    vert0 = vertical_dim(da0)
    if vert0 is None:
        return False, f"{source_var} 缺少气压层维度"
    ref_lat, ref_lon, ref_time = coordinate_names(da0)
    all_levels = []
    for ds in valid:
        da = ds[source_var]
        if vertical_dim(da) != vert0:
            return False, f"{source_var} 的垂直维度名不一致"
        lat, lon, time = coordinate_names(da)
        for left, right, label in ((ref_lat, lat, "latitude"), (ref_lon, lon, "longitude"),
                                   (ref_time, time, "time")):
            if not np.array_equal(da0[left].values, da[right].values):
                return False, f"{source_var} 的 {label} 在各 part 间不一致"
        all_levels.append(pressure_part_levels(ds, source_var))
    union = np.unique(np.concatenate(all_levels)).astype(np.float32)
    desc = np.sort(union)[::-1]
    if len(desc) != len(EXPECTED_PRESSURE_LEVELS) or not np.array_equal(desc, EXPECTED_PRESSURE_LEVELS):
        return False, f"气压层并集异常: {desc.tolist()}"
    return True, ""


def write_pressure_variable(
    datasets: list[xr.Dataset], source_var: str, output: Path, *, variable_name: str,
    units: str, source_dataset: str, grid_resolution: str,
) -> bool:
    """流式写气压层变量:按时次分块 × 逐 part 读取,沿垂直拼成小块,避免整月 4D 物化。"""
    ok, msg = validate_pressure_parts(datasets, source_var)
    if not ok:
        print(f"    [ERROR] pressure {source_var}: {msg}")
        return False

    valid = [ds for ds in datasets if ds is not None and source_var in ds]
    da0 = valid[0][source_var]
    lat_dim, lon_dim, time_dim = coordinate_names(da0)
    R, C, T = (da0.sizes[lat_dim], da0.sizes[lon_dim],
               min(ds[source_var].sizes[time_dim] for ds in valid))
    L = len(EXPECTED_PRESSURE_LEVELS)
    matlab_shape = (R, C, L, T)
    if output.exists() and not OVERWRITE and output_is_valid(output, matlab_shape):
        print(f"    [SKIP] 已存在且已验证: {output}")
        return True
    print(f"    [PLAN] {variable_name}: MATLAB Data={matlab_shape}")
    if DRY_RUN:
        return True

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    if temporary.exists():
        temporary.unlink()
    lat1d, lon1d = da0[lat_dim].values.astype(np.float32), da0[lon_dim].values.astype(np.float32)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    # 按各 part 的最高气压(最大 hPa)降序排列,使逐块沿垂直拼后恰为 1000→10 hPa
    order = sorted(range(len(valid)),
                   key=lambda i: -float(pressure_part_levels(valid[i], source_var).max()))
    try:
        create_mat73_skeleton(
            temporary, lat=lat2d, lon=lon2d, time=to_datenum(da0[time_dim].values),
            level=EXPECTED_PRESSURE_LEVELS, variable_name=variable_name, units=units,
            source_dataset=source_dataset, grid_resolution=grid_resolution, matlab_shape=matlab_shape,
        )
        with h5py.File(temporary, "a") as h5:
            dest = h5["ECMWF/Data"]
            for start in range(0, T, TIME_CHUNK_HOURS):
                stop = min(start + TIME_CHUNK_HOURS, T)
                blocks = []
                for i in order:
                    da = valid[i][source_var]
                    v = vertical_dim(da)
                    block = (da.isel({time_dim: slice(start, stop)})
                             .sortby(v, ascending=False)
                             .transpose(lat_dim, lon_dim, v, time_dim)
                             .values.astype(np.float32, copy=False))
                    blocks.append(block)
                merged = np.concatenate(blocks, axis=2)        # (R, C, 32, t_chunk)
                dest[start:stop, :, :, :] = merged.transpose(3, 2, 1, 0)
                print(f"      [WRITE] {start:04d}:{stop:04d}/{T:04d}")
            if tuple(dest.shape) != matlab_h5_shape(matlab_shape):
                raise RuntimeError(f"写入后 Data 维度异常: {dest.shape}")
            h5.attrs["conversion_status"] = np.bytes_("complete")
        verify_mat73(temporary, matlab_shape)
        os.replace(temporary, output)
        print(f"    [SAVE] {output} ({output.stat().st_size / 1024**3:.2f} GiB)")
        return True
    except Exception as exc:
        print(f"    [ERROR] {output.name} 写入失败: {exc}")
        if temporary.exists():
            temporary.unlink()
        return False


def matlab_h5_shape(matlab_shape: tuple[int, ...]) -> tuple[int, ...]:
    """MATLAB v7.3 文件中 HDF5 Data 的维度顺序与 MATLAB 相反。"""
    return tuple(reversed(matlab_shape))


def output_is_valid(path: Path, matlab_shape: tuple[int, ...]) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            return (
                h5.attrs.get("conversion_status", b"") == b"complete"
                and tuple(h5["ECMWF/Data"].shape) == matlab_h5_shape(matlab_shape)
                and h5["ECMWF/Data"].attrs.get("MATLAB_class", b"") == b"single"
                and all(f"ECMWF/{f}" in h5 for f in ("Time", "Lat", "Lon", "VariableName", "Units"))
            )
    except (OSError, KeyError):
        return False


def create_mat73_skeleton(
    path: Path, *, lat: np.ndarray, lon: np.ndarray, time: np.ndarray,
    level: np.ndarray | None, variable_name: str, units: str,
    source_dataset: str, grid_resolution: str, matlab_shape: tuple[int, ...],
) -> None:
    """用 hdf5storage 建立兼容 MATLAB 的 ECMWF struct，再创建分块 Data。"""
    ecmwf: dict[str, object] = {
        "Data": np.empty((0,), dtype=np.float32),
        "Time": time.astype(np.float64),
        "Lat": lat.astype(np.float32),
        "Lon": lon.astype(np.float32),
        "VariableName": variable_name,
        "Units": units,
        "SourceDataset": source_dataset,
        "GridResolution": grid_resolution,
    }
    if level is not None:
        ecmwf["Level"] = level.astype(np.float32).reshape(-1, 1)
    hdf5storage.savemat(
        str(path), {"ECMWF": ecmwf}, fmt="7.3", store_python_metadata=False,
        appendmat=False,             # 关键：path 可能是 *.tmp，禁止再拼 .mat
        matlab_compatible=True,      # 写出 MATLAB struct 所需 MATLAB_class/MATLAB_fields
        oned_as="column",
        action_for_matlab_incompatible="error",
    )

    shape = matlab_h5_shape(matlab_shape)
    chunk = ((min(TIME_CHUNK_HOURS, shape[0]), min(64, shape[1]), min(64, shape[2]))
             if len(shape) == 3 else
             (min(TIME_CHUNK_HOURS, shape[0]), shape[1], min(64, shape[2]), min(64, shape[3])))
    with h5py.File(path, "a") as h5:
        group = h5["ECMWF"]
        del group["Data"]
        dset = group.create_dataset(
            "Data", shape=shape, dtype=np.float32, chunks=chunk,
            compression="gzip", compression_opts=GZIP_LEVEL, shuffle=True,
        )
        dset.attrs["MATLAB_class"] = np.bytes_("single")
        dset.attrs["H5PATH"] = np.bytes_("/ECMWF")
        h5.attrs["conversion_status"] = np.bytes_("writing")


def stream_data(da: xr.DataArray, path: Path, matlab_shape: tuple[int, ...]) -> None:
    """每次仅从 GRIB 读入 TIME_CHUNK_HOURS 个时次，并写入 HDF5 压缩块。"""
    lat, lon, time = coordinate_names(da)
    vert = vertical_dim(da)
    dims = (lat, lon, time) if vert is None else (lat, lon, vert, time)
    ntime = da.sizes[time]
    with h5py.File(path, "a") as h5:
        destination = h5["ECMWF/Data"]
        for start in range(0, ntime, TIME_CHUNK_HOURS):
            stop = min(start + TIME_CHUNK_HOURS, ntime)
            block = da.isel({time: slice(start, stop)}).transpose(*dims).values.astype(np.float32, copy=False)
            if vert is None:
                destination[start:stop, :, :] = block.transpose(2, 1, 0)
            else:
                destination[start:stop, :, :, :] = block.transpose(3, 2, 1, 0)
            print(f"      [WRITE] {start:04d}:{stop:04d}/{ntime:04d}")
        if tuple(destination.shape) != matlab_h5_shape(matlab_shape):
            raise RuntimeError(f"写入后的 Data 维度异常：{destination.shape}")
        h5.attrs["conversion_status"] = np.bytes_("complete")


def verify_mat73(path: Path, matlab_shape: tuple[int, ...]) -> None:
    """验证结构和首尾压缩块，不把超大 Data 全部读回内存。"""
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("conversion_status", b"") != b"complete":
            raise RuntimeError("文件未完成")
        data = h5["ECMWF/Data"]
        if tuple(data.shape) != matlab_h5_shape(matlab_shape):
            raise RuntimeError(f"Data 维度不符：{data.shape}")
        if data.attrs.get("MATLAB_class", b"") != b"single":
            raise RuntimeError("Data 未标记为 MATLAB single")
        _ = data[0:1]
        _ = data[-1:]
        required = ("Time", "Lat", "Lon", "VariableName", "Units", "SourceDataset", "GridResolution")
        if not all(f"ECMWF/{field}" in h5 for field in required):
            raise RuntimeError("ECMWF 元数据字段不完整")
    print(f"    [VERIFY] MAT v7.3 完整：{path.name}")


def write_variable(
    ds: xr.Dataset, source_var: str, output: Path, *, variable_name: str,
    units: str, source_dataset: str, grid_resolution: str,
) -> bool:
    if source_var not in ds:
        print(f"    [WARN] 未找到变量 {source_var}")
        return False
    da = ds[source_var]
    lat_dim, lon_dim, time_dim = coordinate_names(da)
    vert = vertical_dim(da)
    matlab_shape = ((da.sizes[lat_dim], da.sizes[lon_dim], da.sizes[time_dim]) if vert is None else
                    (da.sizes[lat_dim], da.sizes[lon_dim], da.sizes[vert], da.sizes[time_dim]))
    if output.exists() and not OVERWRITE and output_is_valid(output, matlab_shape):
        print(f"    [SKIP] 已存在且已验证：{output}")
        return True
    print(f"    [PLAN] {variable_name}: MATLAB Data={matlab_shape}")
    if DRY_RUN:
        return True

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    if temporary.exists():
        temporary.unlink()
    lat1d, lon1d = da[lat_dim].values.astype(np.float32), da[lon_dim].values.astype(np.float32)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    try:
        create_mat73_skeleton(
            temporary, lat=lat2d, lon=lon2d, time=to_datenum(da[time_dim].values),
            level=da[vert].values if vert is not None else None,
            variable_name=variable_name, units=units, source_dataset=source_dataset,
            grid_resolution=grid_resolution, matlab_shape=matlab_shape,
        )
        stream_data(da, temporary, matlab_shape)
        verify_mat73(temporary, matlab_shape)
        os.replace(temporary, output)
        print(f"    [SAVE] {output} ({output.stat().st_size / 1024**3:.2f} GiB)")
        return True
    except Exception as exc:
        print(f"    [ERROR] {output.name} 写入失败：{exc}")
        if temporary.exists():
            temporary.unlink()
        return False


def process_dataset(grib: Path, yyyymm: str, mapping: dict[str, tuple[str, str]], out_dir: Path,
                    source_dataset: str, grid_resolution: str) -> bool:
    ds = open_grib(grib)
    if ds is None:
        return False
    ok = True
    # 用显式循环而非 all(...),避免某变量缺失时短路跳过其后的变量
    for src, (name, units) in mapping.items():
        ok &= write_variable(ds, src, out_dir / f"{name}_{yyyymm}.mat", variable_name=name,
                             units=units, source_dataset=source_dataset, grid_resolution=grid_resolution)
    return ok


def main() -> int:
    print("=" * 72)
    print("ERA5 GRIB -> MATLAB v7.3 MAT（按时间流式写入）")
    print(f"输入：{BASE_DIR}\n输出：{OUTPUT_BASE}\nDRY_RUN={DRY_RUN}")
    print("=" * 72)
    success = True

    for grib in sorted(BASE_DIR.glob("era5_land/*/*.grib")):
        print(f"\n[READ] ERA5-Land: {grib.name}")
        success &= process_dataset(grib, grib.parent.name, LAND_VARIABLES,
                                   OUTPUT_BASE / "era5_land" / "0.1deg" / grib.parent.name,
                                   "reanalysis-era5-land", "0.1 degree")

    by_month: dict[str, list[Path]] = defaultdict(list)
    for filename in glob.glob(str(BASE_DIR / "era5_pressure_levels" / "*" / "*.grib")):
        path = Path(filename)
        by_month[path.parent.name].append(path)
    for yyyymm, parts in sorted(by_month.items()):
        print(f"\n[READ] ERA5 pressure levels {yyyymm}: {len(parts)} parts")
        if len(parts) != 4:
            print("    [ERROR] pressure part 数量不是 4，拒绝转换半层数据")
            success = False
            continue
        datasets = [open_grib(p) for p in sorted(parts)]
        if any(ds is None for ds in datasets):
            success = False
            continue
        datasets = [ds for ds in datasets if ds is not None]
        out_dir = OUTPUT_BASE / "era5_pressure_levels" / "0.25deg" / yyyymm
        for src, (name, units) in PRESSURE_VARIABLES.items():
            success &= write_pressure_variable(
                datasets, src, out_dir / f"{name}_{yyyymm}.mat", variable_name=name,
                units=units, source_dataset="reanalysis-era5-pressure-levels",
                grid_resolution="0.25 degree",
            )

    for grib in sorted(BASE_DIR.glob("era5_single_levels/*/*.grib")):
        print(f"\n[READ] ERA5 single levels: {grib.name}")
        success &= process_dataset(grib, grib.parent.name, SINGLE_VARIABLES,
                                   OUTPUT_BASE / "era5_single_levels" / "0.25deg" / grib.parent.name,
                                   "reanalysis-era5-single-levels", "0.25 degree")

    print("=" * 72)
    print("[DONE] 全部转换完成" if success else "[FAILED] 存在失败，请查看 ERROR")
    print("=" * 72)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
