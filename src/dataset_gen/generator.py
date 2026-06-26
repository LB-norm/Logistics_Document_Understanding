from pathlib import Path
from faker import Faker
from weasyprint import HTML
import subprocess
import cv2

from augraphy import AugraphyPipeline
from augraphy import (
    BadPhotoCopy,
    DirtyDrum,
    InkBleed,
    Jpeg,
    LightingGradient,
    Letterpress,
)

fake = Faker("de_DE")
out = Path("synthetic_lieferscheine")
out.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<html>
  <body style="font-family: Arial; font-size: 12px; margin: 24px;">
    <h2>Lieferschein</h2>
    <table style="width:100%; border-collapse: collapse;">
      <tr><td><b>Lieferschein-Nr.</b></td><td>{delivery_no}</td></tr>
      <tr><td><b>Kunde</b></td><td>{customer_name}</td></tr>
      <tr><td><b>Lieferadresse</b></td><td>{address}</td></tr>
      <tr><td><b>Lieferdatum</b></td><td>{date}</td></tr>
      <tr><td><b>Spedition</b></td><td>{carrier}</td></tr>
      <tr><td><b>Packstücke</b></td><td>{packages}</td></tr>
      <tr><td><b>Gesamtgewicht</b></td><td>{weight} kg</td></tr>
    </table>
    <br/>
    <table border="1" cellspacing="0" cellpadding="4" style="width:100%; border-collapse: collapse;">
      <tr>
        <th>Pos</th><th>Artikel-Nr.</th><th>Beschreibung</th><th>Menge</th><th>Einheit</th>
      </tr>
      {rows}
    </table>
    <br/><br/>
    <div>Waren erhalten: _____________________</div>
  </body>
</html>
"""

pipeline = AugraphyPipeline(
    ink_phase=[
        InkBleed(),
        Letterpress(),
    ],
    paper_phase=[
        LightingGradient(),
    ],
    post_phase=[
        DirtyDrum(),
        BadPhotoCopy(),
        Jpeg(quality_range=(25, 60)),
    ],
)

for i in range(100):
    rows = []
    n_items = fake.random_int(min=2, max=12)
    total_weight = 0.0
    for p in range(1, n_items + 1):
        qty = fake.random_int(min=1, max=50)
        weight = round(
            qty
            * fake.pyfloat(
                left_digits=1,
                right_digits=2,
                positive=True,
                min_value=0.1,
                max_value=9.9,
            ),
            2,
        )
        total_weight += weight
        rows.append(
            f"<tr><td>{p}</td><td>{fake.bothify('ART-#####')}</td>"
            f"<td>{fake.word().capitalize()} {fake.word()}</td>"
            f"<td>{qty}</td><td>Stk</td></tr>"
        )

    html = HTML_TEMPLATE.format(
        delivery_no=fake.bothify("LS-########"),
        customer_name=fake.company(),
        address=fake.address().replace("\n", ", "),
        date=fake.date_between(start_date="-2y", end_date="today").strftime("%d.%m.%Y"),
        carrier=fake.company(),
        packages=fake.random_int(min=1, max=20),
        weight=round(total_weight, 2),
        rows="\n".join(rows),
    )

    pdf_path = out / f"ls_{i:05d}.pdf"
    png_prefix = out / f"ls_{i:05d}"

    HTML(string=html).write_pdf(str(pdf_path))

    # PDF -> PNG rendern (hier mit Poppler/pdftoppm)
    subprocess.run(["pdftoppm", "-png", str(pdf_path), str(png_prefix)], check=True)

    img = cv2.imread(str(out / f"ls_{i:05d}-1.png"))
    aug = pipeline(img)
    cv2.imwrite(str(out / f"ls_{i:05d}_scan.png"), aug)
