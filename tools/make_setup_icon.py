"""Draw the installer icon (a 3D toy building brick, 2x2, isometric) as assets/setup_1024.png / assets/setup.icns and
build "Set Up.app" in the project folder, which opens the guided installer in Terminal.
Run: .venv/bin/python tools/make_setup_icon.py"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2048                                     # drawn large, scaled down for smooth edges
C30, S30 = math.cos(math.pi / 6), 0.5
W = D = 2.0                                  # studs across
H = 1.2                                      # brick height (stud pitch = 1)
SR, SH = 0.3, 0.24                           # stud radius / height
SCALE = N * 0.78 / ((W + D) * C30)
CX = N / 2
CY = N / 2 - ((W + D) * S30 * SCALE) / 2 + (H + SH) * SCALE / 2 + N * 0.02

RED = (218, 37, 29)


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def p(x, y, z):
    return (CX + (x - y) * C30 * SCALE, CY + (x + y) * S30 * SCALE - z * SCALE)


def gradient_poly(img, pts, top, bottom):
    """Fill a polygon with a vertical gradient between two colours."""
    ys = [q[1] for q in pts]
    y0, y1 = int(min(ys)), int(max(ys)) + 1
    grad = Image.new("RGBA", img.size)
    gd = ImageDraw.Draw(grad)
    for y in range(y0, y1):
        t = (y - y0) / max(1, y1 - y0)
        gd.line([(0, y), (img.size[0], y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    img.paste(grad, (0, 0), mask)


def stud(img, x, y):
    d = ImageDraw.Draw(img)
    bx, by = p(x, y, H)
    tx, ty = p(x, y, H + SH)
    rx, ry = math.sqrt(2) * C30 * SR * SCALE, math.sqrt(2) * S30 * SR * SCALE
    # side: left half lit, right half in shadow
    side = Image.new("RGBA", img.size)
    sd = ImageDraw.Draw(side)
    for i in range(int(2 * rx) + 3):
        f = 1.0 - 0.38 * (i / (2 * rx))
        sx = bx - rx + i
        sd.line([(sx, ty), (sx, by + ry + 2)], fill=shade(RED, f) + (255,))
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([bx - rx, ty, bx + rx, by], fill=255)
    md.ellipse([bx - rx, by - ry, bx + rx, by + ry], fill=255)
    img.paste(side, (0, 0), mask)
    d.ellipse([tx - rx, ty - ry, tx + rx, ty + ry], fill=shade(RED, 1.18))
    # soft highlight on the stud top
    hl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse([tx - rx * 0.62, ty - ry * 0.75, tx + rx * 0.1, ty + ry * 0.05], fill=(255, 255, 255, 70))
    img.alpha_composite(hl.filter(ImageFilter.GaussianBlur(10)))


def main():
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    # ground shadow
    sh = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    pts = [p(0, 0, 0), p(W, 0, 0), p(W, D, 0), p(0, D, 0)]
    ImageDraw.Draw(sh).polygon([(x + 20, y + 60) for x, y in pts], fill=(0, 0, 0, 110))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(45)))

    top = [p(0, 0, H), p(W, 0, H), p(W, D, H), p(0, D, H)]
    left = [p(0, D, H), p(W, D, H), p(W, D, 0), p(0, D, 0)]
    right = [p(W, 0, H), p(W, D, H), p(W, D, 0), p(W, 0, 0)]
    gradient_poly(img, top, shade(RED, 1.16), shade(RED, 1.05))
    gradient_poly(img, left, shade(RED, 0.92), shade(RED, 0.78))
    gradient_poly(img, right, shade(RED, 0.70), shade(RED, 0.56))
    d = ImageDraw.Draw(img)
    edge = shade(RED, 1.3)
    d.line([p(0, D, H), p(W, D, H), p(W, 0, H)], fill=edge, width=7)   # lit front edges of the top
    d.line([p(W, D, H), p(W, D, 0)], fill=shade(RED, 0.62), width=5)

    for x, y in sorted([(i + 0.5, j + 0.5) for i in range(int(W)) for j in range(int(D))], key=lambda q: q[0] + q[1]):
        stud(img, x, y)

    out = img.resize((1024, 1024), Image.LANCZOS)
    png = os.path.join(ROOT, "assets", "setup_1024.png")
    out.save(png)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "setup.iconset")
        os.mkdir(iconset)
        for size in (16, 32, 128, 256, 512):
            out.resize((size, size), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}.png"))
            out.resize((size * 2, size * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(ROOT, "assets", "setup.icns")], check=True)
    print("wrote assets/setup_1024.png and assets/setup.icns")
    build_app()


APPLESCRIPT = '''set appDir to do shell script "dirname " & quoted form of POSIX path of (path to me)
set setupFile to appDir & "/scripts/mac/setup.command"
try
	do shell script "test -f " & quoted form of setupFile
on error
	display dialog "Set Up can't find the Cautious Giggle files next to it." & return & return & "Keep Set Up inside the Cautious Giggle folder. If you downloaded the folder, move it out of Downloads (for example to your Home folder) and open Set Up again." buttons {"OK"} default button "OK" with icon caution
	return
end try
do shell script "open -a Terminal " & quoted form of setupFile
'''


def build_app():
    import shutil
    app = os.path.join(ROOT, "Set Up.app")
    shutil.rmtree(app, ignore_errors=True)
    subprocess.run(["osacompile", "-o", app, "-e", APPLESCRIPT], check=True)
    shutil.copy(os.path.join(ROOT, "assets", "setup.icns"), os.path.join(app, "Contents", "Resources", "applet.icns"))
    car = os.path.join(app, "Contents", "Resources", "Assets.car")     # newer osacompile icons override applet.icns
    if os.path.exists(car):
        os.remove(car)
    subprocess.run(["/usr/libexec/PlistBuddy", "-c", "Add :CFBundleIdentifier string com.cautiousgiggle.setup",
                    os.path.join(app, "Contents", "Info.plist")], check=False)
    subprocess.run(["/usr/libexec/PlistBuddy", "-c", "Delete :CFBundleIconName",
                    os.path.join(app, "Contents", "Info.plist")], check=False, capture_output=True)
    subprocess.run(["codesign", "--force", "--deep", "-s", "-", app], check=True, capture_output=True)
    os.utime(app)
    print("built Set Up.app")


if __name__ == "__main__":
    main()
