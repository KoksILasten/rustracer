#!/usr/bin/env python3
"""Download CC0 / CC-BY glTF sample scenes for Rustracer.

Sources: Khronos glTF-Sample-Assets (https://github.com/KhronosGroup/glTF-Sample-Assets)
Licenses: DamagedHelmet CC-BY-4.0 (attribution required, see LICENSE.md),
          Lantern and ToyCar CC0-1.0.
"""
import pathlib
import urllib.request

BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/{}/glTF/{}"
LIC = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/{}/LICENSE.md"

SCENES = {
    "DamagedHelmet": [
        "DamagedHelmet.bin", "DamagedHelmet.gltf",
        "Default_AO.jpg", "Default_albedo.jpg", "Default_emissive.jpg",
        "Default_metalRoughness.jpg", "Default_normal.jpg",
    ],
    "Lantern": [
        "Lantern.bin", "Lantern.gltf",
        "Lantern_baseColor.png", "Lantern_emissive.png",
        "Lantern_normal.png", "Lantern_roughnessMetallic.png",
    ],
    "ToyCar": [
        "Fabric_baseColor.png", "Fabric_normal.png", "Fabric_occlusion.png",
        "ToyCar.bin", "ToyCar.gltf",
        "ToyCar_basecolor.png", "ToyCar_clearcoat.png", "ToyCar_emissive.png",
        "ToyCar_normal.png", "ToyCar_occlusion_roughness_metallic.png",
    ],
}

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # assets/scripts -> Rustracer root
total = 0
for name, files in SCENES.items():
    dest = ROOT / "assets" / "scenes" / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        url = BASE.format(name, f)
        out = dest / f
        if out.exists() and out.stat().st_size > 0:
            print(f"skip  {name}/{f}")
            total += out.stat().st_size
            continue
        urllib.request.urlretrieve(url, out)
        size = out.stat().st_size
        total += size
        print(f"save  {name}/{f} ({size} bytes)")
    lic = dest / "LICENSE.md"
    urllib.request.urlretrieve(LIC.format(name), lic)
    print(f"save  {name}/LICENSE.md")
print(f"TOTAL {total} bytes")
