import cdsapi
import os
import time

# ============================================================
# 合并版 ERA5 下载脚本
# 由 download_era5_Land_01degree.py 与 download_era5_0.25degree.py 合并而来
# 统一目录结构、重试逻辑、代理设置与时间配置
# ============================================================

# 创建 CDS API 客户端
c = cdsapi.Client()
# 绕过系统代理（Clash 127.0.0.1:7897 不稳定，且 CDS 可直连），改为直连下载
#c.session.trust_env = False

# ============================================================
# 配置参数（改这里即可）
# ============================================================
BASE_DIR = r"D:/era5_data/"           # 统一根目录

# 起止年月（首尾均包含，改这 4 个数即可）
start_year, start_month = 2022, 7
end_year,   end_month   = 2023, 5

# 数据集开关（需要时单独关闭）
DOWNLOAD_LAND     = True   # ERA5-Land（0.1°）：五项近地面变量
DOWNLOAD_PRESSURE = True   # ERA5 气压层：relative_humidity / temperature / geopotential
DOWNLOAD_SINGLE   = True   # ERA5 单层：geopotential / total_column_water_vapour / boundary_layer_height

# 是否只预演（True 时仅打印目标路径，不真正请求）
DRY_RUN = False

# 下载重试参数
MAX_ATTEMPTS = 5      # 每个请求最多尝试次数
RETRY_DELAY  = 300    # 普通失败等待 5 分钟；CDS 排队限制需要更长时间释放

# 通用区域 [北, 西, 南, 东]
AREA = [54, 73, 3, 135]

# 气压层分批大小（避免单请求过大导致长时间 running 或超时）
BATCH_SIZE = 8

# ============================================================
# 通用常量
# ============================================================
days = ['01', '02', '03', '04', '05', '06', '07',
        '08', '09', '10', '11', '12', '13', '14',
        '15', '16', '17', '18', '19', '20', '21',
        '22', '23', '24', '25', '26', '27', '28',
        '29', '30', '31']

times = ['00:00', '01:00', '02:00', '03:00',
         '04:00', '05:00', '06:00', '07:00',
         '08:00', '09:00', '10:00', '11:00',
         '12:00', '13:00', '14:00', '15:00',
         '16:00', '17:00', '18:00', '19:00',
         '20:00', '21:00', '22:00', '23:00']

# ERA5 气压层（10~1000 hPa，覆盖近地面至 ~31 km 高空）
pressure_levels = [
    '10', '20', '30', '50', '70',
    '100', '125', '150', '175', '200', '225', '250', '300', '350',
    '400', '450', '500', '550', '600', '650', '700', '750', '775',
    '800', '825', '850', '875', '900', '925', '950', '975', '1000',
]
level_batches = [pressure_levels[i:i + BATCH_SIZE]
                 for i in range(0, len(pressure_levels), BATCH_SIZE)]


# ============================================================
# 工具函数
# ============================================================
def year_months(start_year, start_month, end_year, end_month):
    """生成连续的 (年, 月) 字符串列表，如 [('2022','06'), ('2022','07'), ...]"""
    result = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        result.append((f"{y:04d}", f"{m:02d}"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


def target_path(dataset_folder, year, month, filename):
    """构造 D:/era5_data/<数据集>/<YYYYMM>/<filename> 完整路径（不创建目录）。"""
    return os.path.join(BASE_DIR, dataset_folder, f"{year}{month}", filename)


def download(dataset, request, target):
    """带重试的下载：先下到 .tmp，成功后重命名为最终文件（避免留下半截文件）。"""
    if os.path.exists(target):
        if _looks_like_grib(target):
            print(f"⏭️  已存在，跳过：{target}")
            return True
        print(f"⚠️  文件不是有效 GRIB，重新下载：{target}")
        os.remove(target)

    if DRY_RUN:
        print(f"🧪 [DRY_RUN] 将下载到：{target}")
        return True

    os.makedirs(os.path.dirname(target), exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        tmp = target + ".tmp"
        try:
            c.retrieve(dataset, request, tmp)
            if not _looks_like_grib(tmp):
                raise ValueError("CDS 返回的文件不是 GRIB（可能是 ZIP 压缩包）")
            os.replace(tmp, target)
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"✅ 下载完成：{target}（{size_mb:.1f} MB）")
            return True
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            message = str(e)
            print(f"❌ 第 {attempt}/{MAX_ATTEMPTS} 次失败：{message}")
            if attempt < MAX_ATTEMPTS:
                # CDS 对单个数据集的排队请求数有限；快速重试会继续被拒绝。
                if 'queued requests' in message.lower() or 'temporarily limited' in message.lower():
                    delay = max(RETRY_DELAY, 600) * (2 ** (attempt - 1))
                    print(f"   CDS 队列已满，{delay} 秒后再重试（请确保只运行一个下载进程）...")
                else:
                    delay = RETRY_DELAY
                    print(f"   {delay} 秒后重试...")
                time.sleep(delay)
    return False


def _looks_like_grib(path):
    """检查文件头是否包含 GRIB 标识；ZIP 文件头 PK 不会被误认为 GRIB。"""
    try:
        with open(path, 'rb') as f:
            header = f.read(16)
        return header[:4] == b'GRIB'
    except (OSError, PermissionError):
        return False


# ============================================================
# 主程序
# ============================================================
for year, month in year_months(start_year, start_month, end_year, end_month):

    # ============================================================
    # ① ERA5-Land（0.1°）：五项近地面变量
    # ============================================================
    if DOWNLOAD_LAND:
        land_file = target_path(
            "era5_land", year, month, f"era5_land_{year}_{month}.grib")
        print(f"⬇️  正在下载（ERA5-Land 近地面五项）：{year} 年 {month} 月 ...")
        download(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    '2m_dewpoint_temperature',
                    'surface_pressure',
                ],
                'year': year,
                'month': month,
                'day': days,
                'time': times,
                'area': AREA,
                # ERA5-Land 原生分辨率就是 0.1°，不指定 grid，避免额外重网格和请求被拒绝。
                'data_format': 'grib',
                'download_format': 'unarchived',
            },
            land_file
        )

    # ============================================================
    # ② ERA5 气压层：相对湿度、温度（分批下载）
    # ============================================================
    if DOWNLOAD_PRESSURE:
        for bi, batch in enumerate(level_batches, start=1):
            pl_file = target_path(
                "era5_pressure_levels", year, month,
                f"era5_pressure_{year}_{month}_part{bi:02d}.grib")
            print(f"⬇️  正在下载（气压层 第 {bi}/{len(level_batches)} 批）："
                  f"{year} 年 {month} 月 ...")
            download(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        'relative_humidity',   # 相对湿度
                        'temperature',         # 温度
                        'geopotential',        # 各气压层位势（可直接 z/g0 得每层高度，替代/校验测高积分）
                    ],
                    'pressure_level': batch,
                    'year' : year,
                    'month': month,
                    'day'  : days,
                    'time' : times,
                    'area' : AREA,             # [北, 西, 南, 东]
                    'data_format': 'grib',
                    'download_format': 'unarchived',
                },
                pl_file
            )

    # ============================================================
    # ③ ERA5 单层（ERA5 hourly single levels）：位势、整层水汽含量、边界层高度
    # ============================================================
    if DOWNLOAD_SINGLE:
        sl_file = target_path(
            "era5_single_levels", year, month, f"era5_single_{year}_{month}.grib")
        print(f"⬇️  正在下载（单层）：{year} 年 {month} 月 ...")
        download(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential',             # 位势（不要从 ERA5-Land 读取）
                    'total_column_water_vapour',  # 整层可降水量（总水汽含量）
                    'boundary_layer_height',      # 边界层高度
                ],
                'year' : year,
                'month': month,
                'day'  : days,
                'time' : times,
                'area' : AREA,             # [北, 西, 南, 东]
                'data_format': 'grib',
                'download_format': 'unarchived',
            },
            sl_file
        )

print("\n🎉 全部任务执行完毕！")
