"""Generate QR code images for MechanicsDSL resources."""
import qrcode
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "qr_codes"
OUT.mkdir(exist_ok=True)

RESOURCES = {
    "qr_pypi": ("https://pypi.org/project/mechanicsdsl-core/", "PyPI Package"),
    "qr_docs": ("https://mechanicsdsl.readthedocs.io/en/latest/", "Documentation"),
    "qr_github": ("https://github.com/MechanicsDSL/mechanicsdsl", "GitHub Repository"),
    "qr_zenodo": ("https://doi.org/10.5281/zenodo.17771040", "Zenodo DOI Archive"),
}

for filename, (url, label) in RESOURCES.items():
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=20,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    path = OUT / f"{filename}.png"
    img.save(str(path))
    print(f"Saved: {path}  ({label})")

print("\nDone — all QR codes saved to", OUT)
