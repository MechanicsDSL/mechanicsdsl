"""Generate a publication-quality choropleth world map of MechanicsDSL downloads."""
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "bq-results-20260205-003617-1770251816000.xlsx"
OUT  = ROOT / "world_map_downloads.pdf"
OUT_PNG = ROOT / "world_map_downloads.png"

# ── Load data ────────────────────────────────────────────────
wb = openpyxl.load_workbook(XLSX, read_only=True)
ws = wb.active
rows = list(ws.iter_rows(min_row=2, values_only=True))
wb.close()

# country_code is column index 1
country_counts = {}
for row in rows:
    cc = row[1]
    if cc and cc != "None":
        country_counts[cc] = country_counts.get(cc, 0) + 1

print(f"Total download events: {sum(country_counts.values())}")
print(f"Countries: {len(country_counts)}")
for cc, n in sorted(country_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {cc}: {n}")

# ── Map ISO-2 to ISO-3 for geopandas ────────────────────────
# geopandas naturalearth uses ISO_A3
import json

# Manual mapping for the countries we have
ISO2_TO_ISO3 = {
    "US": "USA", "CN": "CHN", "HK": "HKG", "FR": "FRA", "JP": "JPN",
    "DE": "DEU", "RU": "RUS", "SG": "SGP", "KR": "KOR", "IN": "IND",
    "GB": "GBR", "CA": "CAN", "AU": "AUS", "NL": "NLD", "BR": "BRA",
    "SE": "SWE", "IE": "IRL", "TW": "TWN", "IT": "ITA", "PL": "POL",
    "ES": "ESP", "CH": "CHE", "AT": "AUT", "DK": "DNK", "NO": "NOR",
    "FI": "FIN", "IL": "ISR", "CZ": "CZE", "PT": "PRT", "BE": "BEL",
    "MX": "MEX", "ZA": "ZAF", "TH": "THA", "AR": "ARG", "CL": "CHL",
    "NZ": "NZL", "RO": "ROU", "UA": "UKR", "PH": "PHL", "MY": "MYS",
    "ID": "IDN", "TR": "TUR", "VN": "VNM", "CO": "COL", "EG": "EGY",
    "PK": "PAK", "BD": "BGD", "GR": "GRC", "HU": "HUN", "NG": "NGA",
    "PE": "PER", "LT": "LTU", "LV": "LVA", "EE": "EST", "SK": "SVK",
    "SI": "SVN", "BG": "BGR", "HR": "HRV", "RS": "SRB", "KE": "KEN",
    "GH": "GHA", "MA": "MAR", "TN": "TUN", "SA": "SAU", "AE": "ARE",
    "QA": "QAT", "LK": "LKA", "MM": "MMR", "KZ": "KAZ", "UZ": "UZB",
    "GE": "GEO", "AM": "ARM", "AZ": "AZE", "BY": "BLR", "MD": "MDA",
    "BA": "BIH", "MK": "MKD", "AL": "ALB", "ME": "MNE", "CY": "CYP",
    "LU": "LUX", "MT": "MLT", "IS": "ISL", "JO": "JOR", "LB": "LBN",
    "IQ": "IRQ", "IR": "IRN", "AF": "AFG", "NP": "NPL", "EC": "ECU",
    "VE": "VEN", "UY": "URY", "PY": "PRY", "BO": "BOL", "CR": "CRI",
    "PA": "PAN", "GT": "GTM", "DO": "DOM", "PR": "PRI", "CU": "CUB",
    "GF": "GUF", "LI": "LIE", "KH": "KHM", "DZ": "DZA", "IM": "IMN",
}

country_counts_iso3 = {}
for cc, n in country_counts.items():
    iso3 = ISO2_TO_ISO3.get(cc)
    if iso3:
        country_counts_iso3[iso3] = n
    else:
        print(f"  Warning: no ISO3 mapping for {cc}")

# ── Build map ────────────────────────────────────────────────
SHAPEFILE = Path(__file__).resolve().parent / "ne_110m.zip"
world = gpd.read_file(f"zip://{SHAPEFILE}")

world["downloads"] = world["ISO_A3"].map(country_counts_iso3).fillna(0)

# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Custom colormap: white → light blue → dark blue
colors_list = ["#f0f0f0", "#d4e6f1", "#7fb3d8", "#2980b9", "#1a5276"]
cmap = mcolors.LinearSegmentedColormap.from_list("downloads", colors_list, N=256)

# Countries with no downloads: light gray
world[world["downloads"] == 0].plot(
    ax=ax, color="#e8e8e8", edgecolor="#cccccc", linewidth=0.3
)

# Countries with downloads: blue gradient (log scale)
has_downloads = world[world["downloads"] > 0].copy()
has_downloads["log_downloads"] = np.log10(has_downloads["downloads"] + 1)

has_downloads.plot(
    column="log_downloads",
    ax=ax,
    cmap=cmap,
    edgecolor="#999999",
    linewidth=0.4,
    vmin=0,
    vmax=np.log10(max(country_counts.values()) + 1),
)

# Colorbar
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=mcolors.Normalize(vmin=0, vmax=np.log10(max(country_counts.values()) + 1))
)
sm._A = []
cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, aspect=30)
# Set tick labels to actual download counts
max_val = max(country_counts.values())
tick_vals = [1, 10, 100, 1000, max_val]
tick_vals = [v for v in tick_vals if v <= max_val]
cbar.set_ticks([np.log10(v + 1) for v in tick_vals])
cbar.set_ticklabels([str(v) for v in tick_vals])
cbar.set_label("Download events", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=8)

# Title
ax.set_title(
    "MechanicsDSL Global Downloads — 53 Countries",
    fontsize=16, fontweight="bold", pad=12, color="#1a1a1a"
)
ax.text(
    0.5, -0.02,
    f"Source: Google BigQuery · PyPI download logs · {sum(country_counts.values()):,} events · Nov 2025 – Feb 2026",
    transform=ax.transAxes, ha="center", fontsize=8, color="#777777"
)

ax.set_xlim(-180, 180)
ax.set_ylim(-60, 85)
ax.axis("off")

plt.tight_layout()
fig.savefig(str(OUT), bbox_inches="tight", pad_inches=0.2)
fig.savefig(str(OUT_PNG), bbox_inches="tight", pad_inches=0.2, dpi=300)
print(f"\nSaved: {OUT}")
print(f"Saved: {OUT_PNG}")
plt.close()
