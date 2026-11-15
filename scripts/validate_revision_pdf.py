"""Render every final manuscript page and check text/citation integrity."""

import hashlib
import json
from pathlib import Path
import shutil
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw


def main():
    source = Path("paper/main.pdf")
    reader = PdfReader(source)
    texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(texts)
    assert "??" not in text, "Unresolved manuscript cross-reference/citation"
    assert "Ankit Guin" in text
    assert "SVHN" in text and "420" in text
    assert all(number in text for number in ["0.221","0.983","0.294"])
    assert "v2026.11.15-v2" in text
    output = Path("reports/selection_revision/pdf_validation")
    output.mkdir(parents=True,exist_ok=True)
    document = pdfium.PdfDocument(source)
    thumbs = []
    for i,page in enumerate(document):
        bitmap = page.render(scale=1.3)
        picture = bitmap.to_pil().convert("RGB")
        picture.save(output/f"page-{i+1:02d}.png")
        thumb = ImageOps.contain(picture,(265,375))
        canvas = Image.new("RGB",(285,405),"#e6e6e6")
        canvas.paste(thumb,((285-thumb.width)//2,8))
        ImageDraw.Draw(canvas).text((10,385),f"Page {i+1}",fill="black")
        thumbs.append(canvas)
    # Contact sheets keep page-level spacing, float flow and blank regions reviewable.
    for start in range(0,len(thumbs),12):
        subset = thumbs[start:start+12]
        sheet = Image.new("RGB",(285*3,405*((len(subset)+2)//3)),"white")
        for j,thumb in enumerate(subset):
            sheet.paste(thumb,((j%3)*285,(j//3)*405))
        sheet.save(output/f"contact-{start//12+1}.png")
    final = Path("output/pdf/ankit_guin_patch_selection_revision.pdf")
    final.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,final)
    record = dict(pages=len(reader.pages),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  author="Ankit Guin",unresolved_reference_markers=0,all_pages_rendered=True,
                  public_version="v2026.11.15-v2",final_pdf=final.as_posix(),
                  page_text_lengths=[len(t) for t in texts],
                  visual_inspection="pending manual image inspection")
    (output.parent/"manuscript_validation.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
